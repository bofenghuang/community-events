## Todo

- https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=common_voice_11_0&only_verified=0&task=automatic-speech-recognition&config=th&split=train%2Bvalidation&metric=wer
- ESB paper
- convert hf to openai version
    - try saving the state dict of your HF model: torch.save(model.state_dict(), FINE_TUNED_MODEL)
    - https://github.com/luigisaetta/whisper-app/blob/main/match_layers.py
- decoding strategies
- shallow fusion w/ lm
- rescoring nbest with ngram
