from matplotlib import pyplot as plt
import tensorflow as tf


class VGG11Trainer():
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.epochs = []
        self.writer = tf.summary.create_file_writer(self.config.exp.log_dir)

    def metric_names(self):
        return ["loss", "acc"]

    def train(self):
        for epoch in range(self.config.trainer.num_epochs):
            self.epochs.append(epoch)
            self.train_dataset.shuffle(1000)
            self.test_dataset.shuffle(100)
            train_loss = 0
            train_acc = 0
            num_steps = 0
            for image_train, label_train in self.train_dataset:
                loss = self.model.train_on_batch(image_train, label_train)
                print(f"Epoch:{epoch} Step:{num_steps} train_loss: {loss[0]}, train_acc: {loss[1]}")
                train_loss += loss[0]
                train_acc += loss[1]
                num_steps += 1
            train_loss /= num_steps
            train_acc /= num_steps
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            test_loss = 0
            test_acc = 0
            num_steps = 0
            for image_test, label_test in self.test_dataset:
                loss = self.model.test_on_batch(image_test, label_test)
                print(f"Epoch:{epoch} Step:{num_steps} test_loss: {loss[0]}, test_acc: {loss[1]}")
                test_loss += loss[0]
                test_acc += loss[1]
                num_steps += 1
            test_loss /= num_steps
            test_acc /= num_steps
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)

            with self.writer.as_default():
                tf.summary.scalar("loss/train", train_loss, epoch)
                tf.summary.scalar("acc/train", train_acc, epoch)
                tf.summary.scalar("loss/test", test_loss, epoch)
                tf.summary.scalar("acc/test", test_acc, epoch)


        plt.figure(0)
        plt.plot(self.epochs, self.train_losses)
        plt.xlabel('epochs')
        plt.title(f"{self.config.exp.name}: epoch_vs_train loss")
        plt.savefig(f"{self.config.exp.plots_dir}/epoch_vs_train_loss")
        plt.clf()

        plt.plot(self.epochs, self.train_accs)
        plt.xlabel('epochs')
        plt.title(f"{self.config.exp.name}: epoch_vs_train acc")
        plt.savefig(f"{self.config.exp.plots_dir}/epoch_vs_train_accuracy")
        plt.clf()

        plt.plot(self.epochs, self.test_losses)
        plt.xlabel('epochs')
        plt.title(f"{self.config.exp.name}: epoch_vs_test loss")
        plt.savefig(f"{self.config.exp.plots_dir}/epoch_vs_test_loss")
        plt.clf()

        plt.plot(self.epochs, self.test_accs)
        plt.xlabel('epochs')
        plt.title(f"{self.config.exp.name}: epoch_vs_test accuracy")
        plt.savefig(f"{self.config.exp.plots_dir}/epoch_vs_test_accuracy")
