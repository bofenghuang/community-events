## Todo
- normalization script, two versions w or w/ puncs+cases, two trainings
    - make sure have the same features, and consistent across datasets
    - basictextNormalizer

- ESB paper
- regularization, specaugment
    - https://gist.github.com/BirgerMoell/a35c98ff7bca9526b37d2638abca47e2
    - speechbrain
- https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event#recommended-training-configurations
- https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=common_voice_11_0&only_verified=0&task=automatic-speech-recognition&config=th&split=train%2Bvalidation&metric=wer
- standalone evaluation script
- demo https://huggingface.co/spaces/sanchit-gandhi/whisper-small
- https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event#tips-and-tricks
- convert hf to openai version
    - try saving the state dict of your HF model: torch.save(model.state_dict(), FINE_TUNED_MODEL)
    - https://github.com/luigisaetta/whisper-app/blob/main/match_layers.py



Asked Jong Wook Kim (the Whisper author who's speaking today 👀). They did 1-2 runs fine-tuning Whisper on English LibriSpeech and used this training config:
* 16 epochs over the 960h train data
* batch size 256
* linear LR decay from 6.25e-06 to zero
* no weight decay
* no dropout
They had a lot of GPUs to do this! Probably at least an 8x A100. But similar configs should work for us with a bs of 64 and fewer training steps (5-10k should be plenty)
We can also use dropout on smaller datasets to prevent overfitting. Not more than dropout=0.1 though


large-v2: https://github.com/openai/whisper/commit/4179ed2475cc84cba66868b516232ef1b74dacdf


shallow fusion w/ lm 

