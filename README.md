Python Version 3.10.10
pip install --upgrade pip
pip install numpy
pip install tensorflow==2.12.*
pip install argparse
================================================
Verify CPU
================================================
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
===============================================
Verify GPU
===============================================
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
