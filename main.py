import cv2
# هدول قيم تسمى قيم المعنى النموذجي وهي قيم ليست ثابتة (فيني حط اي قيمة لكن يعتبر هدول افضل شي والفائدة منهم التدقيق بالصور )
model_mean_value = (78.4463377603,87.7689143744,114.895847746)

age_list = ['(0, 2)','(4, 6)','(8, 12)',
            '(15, 20)','(25, 32)','(38, 43)',
            '(48, 53)','(60, 100)']

gender_list = ['Male', 'Female']

# دالة استدعاء لملفات تحديد الجنس والعمر
def filesGet():
    age_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_age.prototxt',
        'data/age_net.caffemodel'
    )

    gender_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_gender.prototxt',
        'data/gender_net.caffemodel'
    )

    return (age_net, gender_net)

def read_from_camera(age_net,gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    image = cv2.imread('images/girl2.jpg')
    #هذا الملف خاص بتحديد الوجه
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    # تحديد لون الصورة
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #في حال كان لدينا اكثر من وجه في الصورة الواحدة سوف يحددهم
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if (len(faces)>0): # تحديد عدد الوجوه وطباعته
        print("Found {} Faces".format(str(len(faces))))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y),(x+w, y+h), (255,255,0), 2) # رسم مستطيل
        #جلب الوجه ونسخه وارساله الى الخوارزمية
        face_img = image[y:y+h, h:h+w].copy() 
        # هون بعد ما نسخنا الوجه قلتله كبره وهي 1 وبعدين عطيته القيم المعنى النموذجي واخر وحدة مشان الالوان
        blob = cv2.dnn.blobFromImage(face_img, 1, (227,227), model_mean_value, swapRB=False)
        # توقع الجنس
        gender_net.setInput(blob)
        gender_p= gender_net.forward() # output
        # اما بكون 0 يعني ميل او بكون 1 يعني فيميل
        gender = gender_list[gender_p[0].argmax()]
        print("Gender : " + gender)
         # توقع العمر
        age_net.setInput(blob)
        age_p= age_net.forward() # output
        age = age_list[age_p[0].argmax()]
        print("Age : " + age)
        G_A = "%s %s" % (gender, age)
        cv2.putText(image, G_A, (x, y), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("aya", image)
        cv2.waitKey(0)
# سوف يتم تشغيل الدوال
if __name__ =='__main__':
    age_net, gender_net = filesGet()
    read_from_camera(age_net, gender_net)




