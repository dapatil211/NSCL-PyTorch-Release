# $1 takes in the number of training examples to be used
# $2 takes in the number of epochs
# $3 takes in the GPU ID
# $4 is run name

jac-crun $3 scripts/trainval.py --desc experiments/clevr/desc_nscl_derender.py --training-target derender --curriculum off --dataset clevr_mv --mv --ood-views 2 --data-dir /projects/data/clevr_nscl/multiview_qa/ --val-data-dir /projects/data/clevr_nscl/multiview_qa/ --test-data-dir /projects/data/clevr_nscl/multiview_qa/ --batch-size 32 --epoch $2 --validation-interval 5 --save-interval 5 --data-split 1 --train-split /projects/data/clevr_nscl/multiview_qa/splits/train_$1.json --val-split /projects/data/clevr_nscl/multiview_qa/splits/val.json --test-split /projects/data/clevr_nscl/multiview_qa/splits/test.json --expr $4 --use-tb 1 --resume dumps/clevr_mv/desc_nscl_derender/derender-curriculum_off-qtrans_off-one_shot_1/checkpoints/epoch_10.pth --evaluate
