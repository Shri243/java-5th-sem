from flask import Flask, request, send_file
from PIL import Image, ImageDraw
import numpy as np
import cv2
import numpy as np 
from skimage import color, feature
from scipy.stats import kurtosis
from PIL import Image
import pickle
import io

def extract_features(img):
    # Convert the image to different color spaces
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split the channels
    b, g, r,x = cv2.split(img)
    h, s, v = cv2.split(hsv_img)
    l, a, b_human = cv2.split(lab_img)

    # Calculate mean values
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    mean_h, mean_s, mean_v = np.mean(h), np.mean(s), np.mean(v)
    mean_l, mean_a, mean_bhuman = np.mean(l), np.mean(a), np.mean(b_human)

    # Calculate standard deviations
    std_r, std_g, std_b = np.std(r), np.std(g), np.std(b)
    std_h, std_s, std_v = np.std(h), np.std(s), np.std(v)
    std_l, std_a, std_bhuman = np.std(l), np.std(a), np.std(b_human)

    # Calculate variances
    var_r, var_g, var_b = np.var(r), np.var(g), np.var(b)
    var_h, var_s, var_v = np.var(h), np.var(s), np.var(v)
    var_l, var_a, var_bhuman = np.var(l), np.var(a), np.var(b_human)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kurtosis_r, kurtosis_g, kurtosis_b = kurtosis(r), kurtosis(g), kurtosis(b)
    kurtosis_r_single = np.mean(kurtosis_r)
    kurtosis_g_single = np.mean(kurtosis_g)
    kurtosis_b_single = np.mean(kurtosis_b)
    # Other texture features
    glcm = feature.graycomatrix(gray_img, [1], [0], symmetric=True, normed=True)
    energy = feature.graycoprops(glcm, prop='energy')[0, 0]
    ASM = feature.graycoprops(glcm, prop='ASM')[0, 0]
    contrast = feature.graycoprops(glcm, prop='contrast')[0, 0]
    homogeneity = feature.graycoprops(glcm, prop='homogeneity')[0, 0]

    features_dict = {
        "mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b,
        "mean_h": mean_h, "mean_s": mean_s, "mean_v": mean_v,
        "mean_l": mean_l, "mean_a": mean_a, "mean_bhuman": mean_bhuman,
        "std_r": std_r, "std_g": std_g, "std_b": std_b,
        "std_h": std_h, "std_s": std_s, "std_v": std_v,
        "std_l": std_l, "std_a": std_a, "std_bhuman": std_bhuman,
        "var_r": var_r, "var_g": var_g, "var_b": var_b,
        "var_h": var_h, "var_s": var_s, "var_v": var_v,
        "var_l": var_l, "var_a": var_a, "var_bhuman": var_bhuman,
        "energy": energy, "ASM": ASM, "contrast": contrast, "homogeneity": homogeneity,
        "kurtosis_r_single": kurtosis_r_single, "kurtosis_g_single": kurtosis_g_single, "kurtosis_b_single": kurtosis_b_single
    }
    
    return features_dict

app = Flask(__name__)

def extract_features_and_color(image_path, model):
    large_image = image_path
    width, height = large_image.size
    tile_size = 32

    img_draw = large_image.copy()
    draw = ImageDraw.Draw(img_draw)

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = large_image.crop((x, y, x + tile_size, y + tile_size))
            tile_np = np.array(tile)
            thickness = 2
            features = extract_features(tile_np)
            prediction = model.predict(np.array([list(features.values())]))
            print(prediction)
            #arrr.append(prediction)
            color = (255, 0, 0) if prediction[0] > 0.8 else (0, 255, 0)
            for i in range(thickness):
                draw.rectangle([x - i, y - i, x + tile_size + i, y + tile_size + i], outline=color)
    
    return img_draw

@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = image_file = request.files['image-input']
    print(image_file)
    model = None
    with open('./part2/TreeModel.pkl', 'rb') as file:
        model = pickle.load(file)  # You need to implement a function to load your trained model
    #output_folder_path = 'output.jpg'  # You can customize the output path
    image = Image.open(image_file)
    x = extract_features_and_color(image, model)
    processed_image = x
    processed_image.save('outout.png')
    image_io = io.BytesIO()
    processed_image.save(image_io, format='PNG')
    image_io.seek(0)

    return send_file(image_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
