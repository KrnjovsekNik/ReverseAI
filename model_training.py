
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
"""
def test_model(model_path, test_image_path, conf_threshold=0.5, iou_threshold=0.45, save_path=None):

    if not os.path.exists(model_path):
        print(f"NAPAKA: Model na poti {model_path} ne obstaja!")
        return
    
    if not os.path.exists(test_image_path):
        print(f"NAPAKA: Testna slika na poti {test_image_path} ne obstaja!")
        return
    
    try:
        model = YOLO(model_path)
        print(f"Model uspešno naložen iz: {model_path}")
    except Exception as e:
        print(f"Napaka pri nalaganju modela: {e}")
        return
    
    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"NAPAKA: Slika na poti {test_image_path} ni bila pravilno naložena!")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Slika uspešno naložena iz: {test_image_path}")
    except Exception as e:
        print(f"Napaka pri nalaganju slike: {e}")
        return
    
    try:
        results = model.predict(
            source=image,
            conf=conf_threshold,      
            iou=iou_threshold,        
            max_det=10,               
            verbose=False
        )[0]
        print(f"Detekcija uspešno izvedena. Najdenih {len(results.boxes)} objektov.")
    except Exception as e:
        print(f"Napaka pri izvajanju detekcije: {e}")
        return
    
    output_image = image.copy()

    class_names = ['oseba_zelo_blizu', 'oseba_blizu', 'oseba_dalec', 
                  'vozilo_zelo_blizu', 'vozilo_blizu', 'vozilo_dalec', 
                  'ostalo_zelo_blizu', 'ostalo_blizu', 'ostalo_dalec']
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
             (255, 255, 0), (255, 0, 255), (0, 255, 255),
             (128, 128, 0), (128, 0, 128), (0, 128, 128)]
    
    print("\nZaznani objekti:")
    
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        
        print(f"Detekcija {i+1}: {class_names[cls_id]}, Zaupanje: {conf:.2f}, Koordinate: [{x1}, {y1}, {x2}, {y2}]")
        
        color = colors[cls_id % len(colors)]
        
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls_id]} {conf:.2f}"
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(output_image)
    plt.axis('off')
    plt.title("Detekcija objektov z YOLO modelom")
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
        print(f"\nRezultat shranjen v: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    model_path = "yolo_trening/trening_yolo11_1/weights/best_openvino_model"
    test_image_path = "data/images/test/829.png" 
    
    save_result_path = "rezultati/detekcija_rezultat.png"
    
    test_model(
        model_path=model_path,
        test_image_path=test_image_path,
        conf_threshold=0.6,     
        iou_threshold=0.3,      
        save_path=save_result_path
    )
"""
model = YOLO("yolo_trening/trening_yolo11_1/weights/best.pt")
#model = YOLO("yolo_trening/trening_yolo11_1/weights/best_openvino_model")
img = cv2.imread("data/images/test/836.png")

# Warm-up (important!)
for _ in range(5):
    model(img)

# Measure
N = 50
start = time.time()
for _ in range(N):
    model(img)
end = time.time()

avg_ms = (end - start) / N * 1000
print(f"Average inference time: {avg_ms:.2f} ms")
print(f"Approx FPS: {1000 / avg_ms:.1f}")