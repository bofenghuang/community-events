## Todo

- model w punc+case: need to check if already had in raw dataset, train only cv11 ?
- model w/o punc+case: wip

- ESB paper
- regularization, specaugment
    - https://gist.github.com/BirgerMoell/a35c98ff7bca9526b37d2638abca47e2
    - speechbrain
- https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event#recommended-training-configurations
- https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=common_voice_11_0&only_verified=0&task=automatic-speech-recognition&config=th&split=train%2Bvalidation&metric=wer
- demo https://huggingface.co/spaces/sanchit-gandhi/whisper-small
- https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event#tips-and-tricks
- convert hf to openai version
    - try saving the state dict of your HF model: torch.save(model.state_dict(), FINE_TUNED_MODEL)
    - https://github.com/luigisaetta/whisper-app/blob/main/match_layers.py




decoding strategies
shallow fusion w/ lm 
rescoring nbest with ngram