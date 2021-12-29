from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ImagePreprocessor import *
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost:63342",
    "https://kanavdawra.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

H_RECOGNIZER_MODEL = tf.keras.models.load_model('Models/HRecognizer')
vocab = ['[UNK]', 's', 'Z', 't', 'i', 'p', 'H', 'm', 'z', 'g', 'S', 'J', 'N', '`', 'o', 'y', 'e', 'u', 'E', 'M', 'b',
         'v', 'B', 'L', 'V', 'F', 'l', 'I', 'T', 'A', 'U', 'a', 'h', 'R', 'r', '-', 'W', ' ', 'P', 'n', 'c', 'Y', 'O',
         'C', 'K', 'X', 'f', 'G', 'D', 'Q', "'"]
num_to_char = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None, invert=True)


@app.get("/")
async def root():
    return {"message": "ML Models"}


def preprocess_image(data):
    image = decode_image(data)
    image = convert_image_float32(image)
    image = transpose_image(image, [1, 2, 0])
    image = image_resize(image, (200, 50))
    image = transpose_image(image, [1, 0, 2])
    image = flip_left_to_right(image)
    return image


def get_dict_data(image):
    label = tf.constant([99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                         99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                         99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                         99, 99, 99, 99])

    image = tf.expand_dims(image, axis=0)
    label = tf.expand_dims(label, axis=0)

    batch_object = {'image': image, 'label': label}
    return batch_object


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :34]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


@app.post("/Models/HRecognizer/predict/")
async def predict(file: UploadFile = File(...)):
    print('Reading Image...')
    bytes_string = await file.read()
    print('Preprocessing Image...')
    image = preprocess_image(bytes_string)
    print('Getting Batch Object...')
    batch_object = get_dict_data(image)
    print('Getting Prediction...')
    prediction = H_RECOGNIZER_MODEL.predict(batch_object)
    print('Decoding Prediction...')
    prediction = decode_batch_predictions(prediction)
    print('Returning Prediction...')
    return {prediction[0]}


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
