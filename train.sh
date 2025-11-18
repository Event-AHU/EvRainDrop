NAME="yourname" 
if [ ! -d "log_duke" ]; then   
    mkdir "log_duke"  
fi  

echo "log_duke/${NAME}.log_duke is starting! "

CUDA_VISIBLE_DEVICES=3 \
python train_mars_evrd.py DUKE --multi --backbones rwkv --train_split trainval \
> log_duke/${NAME}.log 2>&1 &