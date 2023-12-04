import numpy as np
import matplotlib.pyplot as plt

def index_sequence_to_string(index_sequence):
  return ", ".join([str(x) for x in index_sequence])


def multi_hot_to_text(multi_hot):
  return " ".join(
      [str(int(x)) for x in multi_hot]
  )

def decode(encoder, indices):
  vocabulary = encoder.get_vocabulary()
  strings = [vocabulary[index] for index in indices]
  return " ".join(strings)

def render_history(history):
  plt.title("losses")
  plt.plot(history.history["loss"], label="loss")
  plt.plot(history.history["val_loss"], label="val_loss")
  plt.legend()
  plt.show()
  plt.close()

  plt.title("Accuracies")
  plt.plot(history.history["accuracy"], label="accuracy")
  plt.plot(history.history["val_accuracy"], label="val_accuracy")
  plt.legend()
  plt.show()
  plt.close()


def compare_histories(history_list):
    for training_name, history in history_list.items():
      plt.plot(history["val_accuracy"], label=training_name)
      
    plt.legend()
    plt.title("Comparision of val_accuracy")
    plt.show()

