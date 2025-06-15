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
