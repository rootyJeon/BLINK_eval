#!/usr/bin/env python3
import os
import sys
import json
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ensure we can import process_vision_info
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "qwen-vl-utils", "src"))
from qwen_vl_utils import process_vision_info

prefix = "./blink/images/"
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

def main(
    questions_path: str = "./blink/questions.jsonl",
    answers_path: str = "./blink/answers.jsonl",
):
    # Load model & processor exactly as in gen.py
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    with open(questions_path, "r") as qf, open(answers_path, "w") as af:
        for line in tqdm(qf):
            q = json.loads(line)
            qid = q["question_id"]
            image_files = q.get("image", [])
            question_text = q.get("text", "")

            # build the single-user message
            messages = [
                {
                    "role": "user",
                    "content": (
                        # one dict per image
                        [{"type": "image", "image": prefix + img} for img in image_files]
                        # then the text prompt
                        + [{"type": "text", "text": question_text}]
                    ),
                }
            ]

            # turn messages → model inputs
            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                #videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # generate and decode
            gen_ids = model.generate(**inputs, max_new_tokens=256)
            # trim off the prompt tokens
            gen_ids = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
            ]
            outputs = processor.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            answer_text = outputs[0]

            # write one JSON line
            record = {
                "question_id": qid,
                "text": answer_text
            }
            af.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()