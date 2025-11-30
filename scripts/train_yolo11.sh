export PYTHONPATH=(pwd):$PYTHONPATH
python scripts/train_yolo11.py
# python -u scripts/train_yolo11.py 2>&1 | perl -ne 'print unless /\r/' | tee train.log
# python -u scripts/train_yolo11.py 2>&1 | sed -u 's/\r/\n/g' | awk 'NF' > train.log