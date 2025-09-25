import mlflow
from tensorflow import keras

# --- Definition des parametres de l'experience ---
EPOCHS = 6
BATCH_SIZE = 110
VALIDATION_SPLIT = 0.1
DROPOUT_RATE = 0.2
DENSE_UNITS_LAYER1 = 512
DENSE_UNITS_OUTPUT = 10
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

# Définir l'URI de suivi pour pointer vers votre serveur MLflow local
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("MLflow tracking URI set to: http://127.0.0.1:5000")

# Initialisation de MLflow
with mlflow.start_run():
    #log des parametres
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("validation_split", VALIDATION_SPLIT)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)
    mlflow.log_param("dense_units_layer1", DENSE_UNITS_LAYER1)
    mlflow.log_param("dense_units_output", DENSE_UNITS_OUTPUT)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("loss_function", LOSS_FUNCTION)
    mlflow.log_param("metrics", METRICS)

    # Chargement du jeu de donnees MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalisation des donnees
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Redimensionnement des images pour les reseaux fully-connected
    x_train = x_train.reshape(60000 , 784)
    x_test = x_test.reshape(10000 , 784)
    # Construction du modele
    model = keras.Sequential([
        keras.layers.Dense(DENSE_UNITS_LAYER1, activation ='relu', input_shape =(784, ) ),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(DENSE_UNITS_OUTPUT, activation ='softmax')
    ])
    # Compilation du modele
    model.compile (optimizer =OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)

    # Entrainement du modele
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )

    # valuation du modele
    test_loss , test_acc = model.evaluate(x_test, y_test)
    mlflow.log_metric("test_accuracy", test_acc)
    print (f"Precision sur les donnees de test : { test_acc :.4f}")

    # Sauvegarde du modele
    #model.save("mnist_model.h5")
    #print ("Modele sauvegader sous mnist_model.h5")

    # SAUVEGARDE DU MODELE AVEC MLflow
    mlflow.keras.log_model(model, "mnist_model_keras")
    print("Modele sauvegarde avec MLflow sous le nom de mnist_model_keras")

print("MLflow run termine.")