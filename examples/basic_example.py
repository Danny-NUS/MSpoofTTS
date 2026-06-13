import os

import soundfile as sf

from neutts import NeuTTS


def main(
    input_text,
    ref_audio_path,
    ref_text,
    backbone,
    backbone_device="cpu",
    codec="neuphonic/neucodec",
    codec_device="cpu",
    sampling_scheme="orig",
    discriminator_repo="Chanson-0803/MSpoofTTS",
    output_path="output.wav",
):
    if not ref_audio_path or not ref_text:
        print("No reference audio or text provided.")
        return None

    use_hier = sampling_scheme == "rank_eas_hier"

    # MSpoofTTS only loads discriminator checkpoints when the guided scheme is used.
    tts = NeuTTS(
        backbone_repo=backbone,
        backbone_device=backbone_device,
        codec_repo=codec,
        codec_device=codec_device,
        use_hier=use_hier,
        discriminator_repo=discriminator_repo,
    )

    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()

    print("Encoding reference audio")
    ref_codes = tts.encode_reference(ref_audio_path)

    print(f"Generating audio with sampling_scheme={sampling_scheme}: {input_text}")
    wav = tts.infer(input_text, ref_codes, ref_text, sampling_scheme=sampling_scheme)

    print(f"Saving output to {output_path}")
    sf.write(output_path, wav, 24000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS/MSpoofTTS inference example")
    parser.add_argument(
        "--input_text", type=str, required=True, help="Input text to be converted to speech"
    )
    parser.add_argument(
        "--ref_audio", type=str, default="./samples/jo.wav", help="Path to reference audio file"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="./samples/jo.txt",
        help="Reference text corresponding to the reference audio",
    )
    parser.add_argument(
        "--output_path", type=str, default="output.wav", help="Path to save the output audio"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="neuphonic/neutts-nano",
        help="Hugging Face repo containing the NeuTTS backbone checkpoint",
    )
    parser.add_argument("--backbone_device", type=str, default="cpu", help="Backbone device")
    parser.add_argument(
        "--codec", type=str, default="neuphonic/neucodec", help="Codec repo or local ONNX path"
    )
    parser.add_argument("--codec_device", type=str, default="cpu", help="Codec device")
    parser.add_argument(
        "--sampling_scheme",
        type=str,
        default="orig",
        choices=["orig", "eas", "rank_eas_hier"],
        help="Decoding scheme. Use rank_eas_hier for MSpoofTTS-guided inference.",
    )
    parser.add_argument(
        "--discriminator_repo",
        type=str,
        default="Chanson-0803/MSpoofTTS",
        help="Hugging Face repo containing MSpoofTTS discriminator checkpoints",
    )
    args = parser.parse_args()
    main(
        input_text=args.input_text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        backbone=args.backbone,
        backbone_device=args.backbone_device,
        codec=args.codec,
        codec_device=args.codec_device,
        sampling_scheme=args.sampling_scheme,
        discriminator_repo=args.discriminator_repo,
        output_path=args.output_path,
    )
