from flask import Flask, render_template, request
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import base64
from tensorflow.keras import models

labels = ('character_10_yna',
          'character_11_taamatar',
          'character_12_thaa',
          'character_13_daa',
          'character_14_dhaa',
          'character_15_adna',
          'character_16_tabala',
          'character_17_tha',
          'character_18_da',
          'character_19_dha',
          'character_1_ka',
          'character_20_na',
          'character_21_pa',
          'character_22_pha',
          'character_23_ba',
          'character_24_bha',
          'character_25_ma',
          'character_26_yaw',
          'character_27_ra',
          'character_28_la',
          'character_29_waw',
          'character_2_kha',
          'character_30_motosaw',
          'character_31_petchiryakha',
          'character_32_patalosaw',
          'character_33_ha',
          'character_34_chhya',
          'character_35_tra',
          'character_36_gya',
          'character_3_ga',
          'character_4_gha',
          'character_5_kna',
          'character_6_cha',
          'character_7_chha',
          'character_8_ja',
          'character_9_jha',
          'digit_0',
          'digit_1',
          'digit_2',
          'digit_3',
          'digit_4',
          'digit_5',
          'digit_6',
          'digit_7',
          'digit_8',
          'digit_9')

app = Flask(__name__)


@app.route('/', methods=["POST", "GET"])
def main_page():
    return render_template("MainPage.html")


@app.route('/3', methods=['POST'])
def result_page():
    draw = request.form.get("url")
    draw = draw[21:]
    # Decoding
    draw_decoded = base64.b64decode(draw)
    image = np.asarray(bytearray(draw_decoded), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = image.astype("float") / 255.0
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = img_to_array(image)
    print(image.dtype)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    model = models.load_model('handwriting_model.h5')
    result_list = model.predict(image)
    result = labels[result_list.argmax()]
    return render_template("ResultPage.html", result=result)


if __name__ == '__main__':
    app.run(debug=True)
