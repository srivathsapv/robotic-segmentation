echo "TODO Add fold 0"
for i in 1 2 3
do
   python train.py --fold $i --lr 0.0001 --n-epochs 10 --type parts --jaccard-weight 1
   python train.py --fold $i --lr 0.00001 --n-epochs 20 --type parts --jaccard-weight 1
done
