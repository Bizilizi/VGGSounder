# Model name
# For instance video-llama-2-av
model=$1

# Modality
modality=$2

# Batch size
batch_size=${3:-2}

# Prompt
# Set the appropriate prompt based on the modality
if [ "$modality" = "a" ]; then
    PROMPT="What actions are being performed in this audio, explain all sounds and actions in the audio? Please provide a short answer."
else
    PROMPT="What actions are being performed in this video, explain all sounds and actions in the video? Please provide a short answer."
fi

python evaluate_model.py \
    --model Qwen/Qwen3-32B \
    --question \"$PROMPT\" \
    --prediction_csv \"./csv/$modality/predictions-$model.csv\" \
    --target_csv \"../../vggsounder/data/vggsounder+background-music.csv\" \
    --backend transformers \
    --batch_size $batch_size