import tensorflow as tf

# 1. Load model hiện tại (đang là .keras)
model = tf.keras.models.load_model('models/gold_price_lstm_model.keras')

# 2. Lưu lại dưới dạng .h5 (Định dạng Legacy tương thích mọi phiên bản)
model.save('models/gold_price_lstm_model.h5')

print("Done")