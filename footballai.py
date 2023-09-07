import numpy as np
from tensorflow import keras
from keras import layers
from art import text2art


print("============================================================")
ascii_art = text2art("FootballAi")
print(ascii_art)
print("Football game winner prediction artificial intelligence")
print("============================================================")
print("Created by: Corvus Codex")
print("Github: https://github.com/CorvusCodex/")
print("Licence : MIT License")
print("Support my work:")
print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

data = np.loadtxt('data.txt', delimiter=',', dtype=int)

train_data = data[:int(0.8*len(data))]
val_data = data[int(0.8*len(data)):]

max_value = np.max(data)

model = keras.Sequential()
model.add(layers.Embedding(input_dim=max_value+1, output_dim=64))
model.add(layers.LSTM(256))
model.add(layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100)

predictions = model.predict(val_data)

predicted_winner = np.argmax(predictions, axis=1)

print("============================================================")
print("Predicted Winner:")
for winner in predicted_winner[:1]:
    if winner == 0:
        print('The Predicted Winner is: TEAM 1')
    else:
        print('The Predicted Winner is: TEAM 2')

print("============================================================")
print("Support my work:")
print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

# Prevent the window from closing immediately
input('Press ENTER to exit')

