t2t-decoder \
    --t2t_usr_dir=self_script \
    --problem=my_problem \
    --data_dir=./self_data \
    --model=lstm_seq2seq_attention \
    --hparams_set=lstm_attention \
    --output_dir=./train \
    --decode_hparams="beam_size=4,alpha=0.6" \
    --decode_from_file=decoder/q.txt \
    --decode_to_file=decoder/a.txt


