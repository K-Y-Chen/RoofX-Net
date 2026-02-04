# Rooftopplus
echo python3 train.py --lr 0.007 --workers 4 --epochs 300 --batch-size 24 --expname expname --eval-interval 1 --dataset rooftop --backbone mobilenet --alpha 0.7 --beta 0.3 --no-binary-cls &&
python3 train.py --lr 0.007 --workers 4 --epochs 300 --batch-size 24 --expname expname --eval-interval 1 --dataset rooftop --backbone mobilenet --alpha 0.7 --beta 0.3 --no-binary-cls &&


# WHU
echo python3 train.py --lr 0.007 --workers 4 --epochs 300 --batch-size 24 --expname expname --eval-interval 1 --dataset WHU --backbone mobilenet --alpha 0.7 --beta 0.3 &&
python3 train.py --lr 0.007 --workers 4 --epochs 300 --batch-size 24 --expname expname --eval-interval 1 --dataset WHU --backbone mobilenet --alpha 0.7 --beta 0.3 &&

echo Training done



