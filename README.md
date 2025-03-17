ในส่วนของโค้ดนี้ผมอาจจะไม่ได้ทำเองหมดเเพราะผมยังเรียนรู้ไม่มากพอซึ่งผมนำAIมาช่วยบ้างขออภัยอาจารย์ด้วยนะครับตอนนี้กำลังศึกษาเพิ่มเติมอยู่ครับ 
install

git clone https://github.com/Anusit66022275/alexnet-mnist.git
cd alexnet-mnist

pip install -r requirements.txt


 ดาวน์โหลดไฟล์โมเดล `alexnet_mnist.pth`

เนื่องจาก GitHub ไม่สามารถเก็บไฟล์ใหญ่เกิน 100MB ผมเลยต้อใช้โหลดลิ้งด้านล่าง

 [Click to download alexnet_mnist.pth](https://drive.google.com/file/d/1CfMmGgLNRk70LQX57S2SbLLEeOt2_dC8/view?usp=drive_link)

หลังจากดาวน์โหลดเสร็จ ให้นำไฟล์ไปวางในโฟลเดอร์นี้:
alexnet_ai/models/alexnet_mnist.pth

ถ้าจะรันคือ python alexnet_ai/main.py ครับ
