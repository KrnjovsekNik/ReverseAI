conda create -n ovire_yolo python=3.10 -y
conda activate ovire_yolo
pip install ultralytics opencv-python notebook albumentations chardet charset_normalizer
jupyter-notebook
--> ustvari mapo ReverseAI
cd ReverseAI
--> samo kreator: git init
--> git remote add origin "link"
git add . 
git commit -m "commit_ime"
git push -u origin master
--> za ostale git clone git clone https://github.com/KrnjovsekNik/ReverseAI.git

moreš imet odprt docker app
docker-compose up --build
Da zaženeš kamer client:
python3 camera_mqtt.py
ko končaš
docker-compose down 

git branchi:
git checkout -b nova-funcijonalnost
git add .
git commit -m "opis"
git push origin nova-funcijonalnost
merganje:
git checkout main
git pull origin main
git merge nova-funkcionalnost
git push origin main

Vzpostavitev brezzicnega delovanja.

Nalozi:
  Nalozi si IPWebcam na telefonu
  Nalozi si termux na telefonu LINK!-> https://f-droid.org/packages/com.termux/
  Nalozi kamera.py datoteko

Na telefonu
  Zazeni IPWebcam in nastavi background delovanje
  Zazeni termux
  Zazeni hotspot in povezi laptop z telefonim
  
  Na termux klici: 

    samo prvic:
      termux-setup-storage
      pkg update && pkg upgrade
      pkg install python
      pip install paho-mqtt requests

    vsakic:
      cd /storage/emulated/0/Download/

Na laptopu:
  conda activate ovire_yolo
  zdaj na laptopu v datoteki UI zazenes "docker-compose up --build", da zazenes mqtt server in yolo obdelava server
  na laptopu odpres cmd in klices ipconfig -> preberes svoj IP
  na laptopu odpres nov anaconda prompt, aktiviras env, se premaknes v UI, potem frontend, in klices python user_interface.py

pogledas se v IPWebcam, tisti ip ki ti tam pise
nato v Termux terminalu klices (vse brez narekovajev!) python kamera.py --broker "napises ip na laptopu" --cam_ip "napises ip od IPWebcama z portom vkljucno" --delay "svelika sekund med vsako poslano sliko"








