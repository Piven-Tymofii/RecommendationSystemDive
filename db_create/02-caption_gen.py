'''
Generate captions for audio clips using a quantized Qwen2-Audio model.
Saves results incrementally to a checkpoint CSV. 
Results - captioned data saved to 'captions_checkpoint.csv'.
'''

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import librosa  # for resampling fallback
import soundfile as sf
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import time
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_model_quantized():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct",
                                               trust_remote_code=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,           # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",   # Use NF4 quantization
        bnb_4bit_compute_dtype=torch.float16  # Compute in float16 for speed
    )

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config,
    )
    model.eval()
    return processor, model


# Keep the "max_duration" semantics: we'll pass 15.0 from main
def get_middle_segment(path, max_duration=None, portion=0.30, sr=None):
    """
    Loads the middle segment of the file. Uses soundfile to read only the needed frames 
    when possible, falling back to librosa if needed. Returns mono float32.
    """
    try:
        info = sf.info(path)
        file_sr = info.samplerate
        if sr is None:
            sr = file_sr
        # Compute how many samples to extract from the source file (at file_sr)
        target_samples_in_src = int(min(portion * info.frames, float(max_duration) * file_sr))
        start_frame = max(0, (info.frames - target_samples_in_src) // 2)
        with sf.SoundFile(path) as f:
            f.seek(start_frame)
            data = f.read(frames=target_samples_in_src, dtype='float32', always_2d=True)
            # to mono
            if data.ndim == 2:
                data = data.mean(axis=1)
        # resample if needed
        if file_sr != sr:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        return data.astype('float32'), sr
    except Exception as e:
        # fallback: full load with librosa then slice
        logging.warning(f"soundfile fast-path failed for {path}: {e}. Falling back to librosa.load")
        y, file_sr = librosa.load(path, sr=sr)
        total_samples = len(y)
        max_samples = int(max_duration * file_sr)
        segment_samples = min(int(portion * total_samples), max_samples)
        start = (total_samples - segment_samples) // 2
        end = start + segment_samples
        return y[start:end].astype('float32'), file_sr


def build_prompt_for_audio():
    """Return the prompt text that will be used for every example. We keep audio separate.
    The processor.apply_chat_template still needs the conversation structure, so we construct
    a reusable template piece here."""
    return (
        "Generate a vivid descriptive music caption in English (1-2 sentences). "
        "Then provide 4 short keywords separated by commas. Output format: <caption> || kw1, kw2, kw3, kw4"
    )


def caption_clip(processor, model, audio, sr, max_new_tokens=48):
    conversation = [
        {"role": "system", "content": "You are a helpful music descriptor."},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": build_prompt_for_audio()}
        ]}
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, sampling_rate=sr, return_tensors="pt", padding=True).to(model.device)

    # Valid generation kwargs — note we avoid passing flags that newer transformers warn about
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    with torch.inference_mode():
        # Use the recommended torch.amp.autocast interface to silence the FutureWarning
        if model.device.type == 'cuda':
            # device_type 'cuda' ensures correct behavior on GPUs
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                generated_ids = model.generate(**inputs, **gen_kwargs)
        else:
            generated_ids = model.generate(**inputs, **gen_kwargs)

    generated = generated_ids[0, inputs.input_ids.shape[-1]:]
    caption = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return caption


def caption_batch(processor, model, audios, srs, max_new_tokens=48):
    """Batch a list of audio arrays into a single generate call. Returns list of captions."""
    # Build conversation prompts for each item (audio differs)
    text_prompts = []
    for audio in audios:
        conversation = [
            {"role": "system", "content": "You are a helpful music descriptor."},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": build_prompt_for_audio()}
            ]}
        ]
        text_prompts.append(processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False))

    # sampling_rate should be the same for all items; we assume srs[0] == processor.feature_extractor.sampling_rate
    inputs = processor(text=text_prompts, audio=audios, sampling_rate=srs[0], return_tensors="pt", padding=True).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )

    with torch.inference_mode():
        if model.device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                generated_ids = model.generate(**inputs, **gen_kwargs)
        else:
            generated_ids = model.generate(**inputs, **gen_kwargs)

    captions = []
    for i in range(len(text_prompts)):
        gen = generated_ids[i, inputs.input_ids.shape[-1]:]
        caption = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
        captions.append(caption)
    return captions


def append_checkpoint_rows(checkpoint_csv, rows):
    # rows is list of dicts with TRACK_ID and caption
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    header = not os.path.exists(checkpoint_csv)
    df_new.to_csv(checkpoint_csv, mode='a', header=header, index=False)


def prefetch_segments(rows, processor_sr, max_duration=15.0, portion=0.30, max_workers=4):
    """Prefetch several middle segments in parallel. Returns list of tuples (row, seg, sr)."""
    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(get_middle_segment, row.PATH, max_duration, portion, processor_sr): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                seg, sr = fut.result()
            except Exception as e:
                logging.error(f"Load error for {row.TRACK_ID}: {e}")
                seg, sr = None, None
            out.append((row, seg, sr))
    return out


def main(df_path, checkpoint_csv="captions_checkpoint.csv"):
    df = pd.read_csv(df_path)
    if os.path.exists(checkpoint_csv):
        done = set(pd.read_csv(checkpoint_csv)["TRACK_ID"].tolist())
    else:
        done = set()

    # ensure caption column exists
    if "caption" not in df.columns:
        df["caption"] = ""

    processor, model = init_model_quantized()
    target_sr = processor.feature_extractor.sampling_rate

    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # warm-up a single synthetic clip to avoid first-call overhead
    warm = np.random.randn(int(1.0 * target_sr)).astype('float32')
    try:
        _ = caption_clip(processor, model, warm, sr=target_sr, max_new_tokens=24)
    except Exception:
        pass

    # params you can tune
    batch_size = 2  # try 2; drop to 1 if OOM
    prefetch_workers = 4
    max_duration = 10.0 
    portion = 0.30
    save_every = 50
    max_new_tokens = 55

    rows = [r for r in df.itertuples(index=True) if r.TRACK_ID not in done]

    processed_since_save = 0

    start_all = time.time()
    for i in tqdm(range(0, len(rows), batch_size), total=(len(rows) + batch_size - 1) // batch_size):
        chunk = rows[i:i+batch_size]
        t0 = time.time()
        prefetched = prefetch_segments(chunk, target_sr, max_duration=max_duration, portion=portion, max_workers=prefetch_workers)

        audios = []
        srs = []
        idxs = []
        tids = []

        for row, seg, sr in prefetched:
            if seg is None or len(seg) == 0:
                logging.warning(f"Empty segment for {row.TRACK_ID}, using short silence fallback")
                seg = np.zeros(int(1.0 * target_sr), dtype='float32')
                sr = target_sr
            audios.append(seg.astype('float32'))
            srs.append(sr)
            idxs.append(row.Index)
            tids.append(row.TRACK_ID)

        # try batched generation
        try:
            captions = caption_batch(processor, model, audios, srs, max_new_tokens=max_new_tokens)
        except RuntimeError as e:
            logging.warning(f"OOM or runtime error on batched generation: {e}. Falling back to per-sample generation.")
            torch.cuda.empty_cache()
            captions = []
            for a, sr in zip(audios, srs):
                try:
                    captions.append(caption_clip(processor, model, a, sr=sr, max_new_tokens=max_new_tokens))
                except Exception as e2:
                    logging.error(f"Generation failed for one sample: {e2}")
                    captions.append("")

        # store results
        new_rows = []
        for idx, tid, cap in zip(idxs, tids, captions):
            df.at[idx, "caption"] = cap
            if tid not in done:
                new_rows.append({"TRACK_ID": tid, "caption": cap})
                done.add(tid)

        if new_rows:
            append_checkpoint_rows(checkpoint_csv, new_rows)
            processed_since_save += len(new_rows)

        if processed_since_save >= save_every:
            logging.info(f"Checkpoint saved after {len(done)} captions.")
            processed_since_save = 0

        t1 = time.time()
        logging.info(f"Chunk processed ({len(audios)} items) in {t1 - t0:.2f}s; total done {len(done)}")

    total_time = time.time() - start_all
    logging.info(f"All captions generated and saved. Total time: {total_time/60.0:.2f} minutes")

    # final save of full dataframe subset (optional)
    try:
        df[df["TRACK_ID"].isin(list(done))].to_csv(checkpoint_csv.replace('.csv', '_full.csv'), index=False)
    except Exception as e:
        logging.warning(f"Could not write full CSV snapshot: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True, help="Path to dataframe CSV")
    parser.add_argument("--checkpoint", type=str, default="captions_checkpoint.csv", help="Path to checkpoint CSV")
    args = parser.parse_args()
    main(args.metadata, args.checkpoint)