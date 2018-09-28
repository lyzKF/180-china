USR_DIR=self_script
PROBLEM=my_problems
DATA_DIR=./self_data
OUTPUT_DIR=./train
MODEL=lstm_seq2seq_attention
HPARAMS_SET=lstm_attention

t2t-trainer \
    --t2t_usr_dir=$USR_DIR \
    --problem=$PROBLEM \
    --data_dir=$DATA_DIR \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --output_dir=$OUTPUT_DIR
