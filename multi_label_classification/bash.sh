for i in 1 2 3 4 5
do
	CUDA_VISIBLE_DEVICES=0 python3 train.py -i ${i} -n 60
done

CUDA_VISIBLE_DEVICES=0 python3 predict.py

