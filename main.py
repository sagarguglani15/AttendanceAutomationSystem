import cv2
from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
import datetime
import os
from PIL import ImageTk
from PIL import Image as img_pil
import pymysql as sql
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk as ttk


db=sql.connect('localhost','root','rootpasswordgiven','attendance')
cursor=db.cursor()


df=pd.read_csv("record.csv")
date=datetime.datetime.now().strftime("%d/%m/%Y")
if date not in df.columns:
    df[date]='A'  #add new date column
df.to_csv("record.csv",index=False)


top=Tk()
top.geometry('1200x700')
top.config(background='White')
top.resizable(False, False)
top.title('Attendance Automation System')

def for_stud(id):
    df = pd.read_csv("record.csv")
    l = list(df['Id'])
    if id not in l:
        messagebox.showerror('Record not found', 'Found no attendance record for this student!')
        return
    res = dict(df.loc[list(df.index[df['Id'] == id])[0]])
    p, a = 0, 0
    for i in res.values():
        if i == 'P':
            p += 1
        elif i == 'A':
            a += 1
    figure_bar2 = plt.Figure(figsize=(4, 4))
    a_bar2 = figure_bar2.add_subplot(111)
    a_bar2.pie([p, a], labels=['Present', 'Absent'], explode=[0.1, 0], shadow=True, autopct='%1.1f%%')
    chart_bar2 = FigureCanvasTkAgg(figure_bar2, top)
    chart_bar2.get_tk_widget().place(x=700, y=330)

def update_attendance(id):
    df = pd.read_csv("record.csv")
    l = list(df['Id'])
    if id not in l:
        return
    res = dict(df.loc[list(df.index[df['Id'] == id])[0]])
    p, a = 0, 0
    for i in res.values():
        if i == 'P':
            p += 1
        elif i == 'A':
            a += 1
    figure_bar = plt.Figure(figsize=(4, 4))
    a_bar = figure_bar.add_subplot(111)
    a_bar.pie([p, a], labels=['Present', 'Absent'], explode=[0.1,0], shadow=True, autopct='%1.1f%%')
    chart_bar = FigureCanvasTkAgg(figure_bar, top)
    chart_bar.get_tk_widget().place(x=50, y=330)


def register(id,x):
    df = pd.read_csv("record.csv")

    l=list(df['Id'])
    if id in l:
        messagebox.showinfo('Record Present', 'You have already registered some face data with this ID')
        return
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    result=[]
    try:
        cursor.execute(f'select name, type from Users where id={id}')
        result=cursor.fetchall()
        db.commit()
    except Exception as e:
        db.rollback()
        messagebox.showerror('SQL Error', e)
        return

    name=result[0][0]
    type=result[0][1]
    sampleNum = 0
    while (True):
        ret, img = cam.read()
        if ret==False:
            messagebox.showerror('WebCam not found', 'Please try reconnecting your WebCam!')
            return
        faces = detector.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (144, 1, 1), 3)

            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            cv2.imwrite(
                "face/user." + str(id) + '.' + str(sampleNum) + ".jpg",
                img[y:y + h, x:x + w])
        try:
            cv2.imshow('frame', img)
        except:
            messagebox.showerror('Web Camera Not Found', 'Try reconnecting your camera again !')
            return
        # wait for 100 miliseconds
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 20
        elif sampleNum > 20:
            df2 = pd.DataFrame({"Id": [id], "Name": [name], "Type": [type]})
            df = pd.concat([df, df2]).drop_duplicates().reset_index(drop=True)
            df.to_csv("record.csv", index=False)
            messagebox.showinfo('Done', 'Face Registered Succesfully!')
            break
    cam.release()
    cv2.destroyAllWindows()
    

def train(x):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = 'face'

    def getImagesWithID(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        IDs = []

        for imagePath in imagePaths:
            faceImg = img_pil.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')
            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow("training", faceNp)
            cv2.waitKey(100)  # for 10 milliseconds wait......
        return np.array(IDs), faces

    Ids, faces = getImagesWithID(path)
    recognizer.train(faces, np.array(Ids))
    recognizer.save('trainningData.yml')
    cv2.destroyAllWindows()

def attend(nid,x):
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("trainningData.yml")
    id=0
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    df = pd.read_csv("record.csv")

    while (True):

        ret, img = cam.read()
        if ret==False:
            messagebox.showerror('WebCam not found', 'Please try reconnecting your WebCam!')
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        faces = faceDetect.detectMultiScale(gray, 2, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            id, conf = rec.predict(gray[y:y + h, x:x + w])
            id1 = df['Name'][df.Id == id]
            df[date][df.Id == id] = 'P'
            df.to_csv("record.csv", index=False)
            cv2.putText(img, str(id1), (x, y + h), font, 3, 255)
        try:
            cv2.imshow("Face", img)
        except:
            messagebox.showerror('Web Camera Not Found', 'Try reconnecting your camera again !')
            return
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
    print(nid)
    update_attendance(nid)
    cam.release()
    cv2.destroyAllWindows()

def student(id):
    children = top.winfo_children()
    for i in range(2,len(children)):
        children[i].place_forget()

    Label(top, text=f'Student ID: {id}', bg='white', fg='black', font=('ABC 20 bold italic')).place(x=620, y=100)
    result = []
    try:
        cursor.execute(f'select name from Users where id={id}')
        result = cursor.fetchall()
        db.commit()
    except Exception as e:
        db.rollback()
        messagebox.showerror('SQL Error', e)
        return

    Label(top,text=f'{result[0][0]}', bg='white', fg='black', font=('ABC 20 bold')).place(x=970,y=100)

    ctrain=Canvas(width=160, height=180, highlightthickness=0, background='midnightblue')
    timage=img_pil.open('train_system1.png')
    tkimage=ImageTk.PhotoImage(timage)
    top.tkimage=tkimage
    ctrain.create_image(5, 5, image=tkimage, anchor='nw')
    ctrain.create_text(5, 160, text='Train your System', fill='white', font='ABc 11 bold', anchor='nw')
    ctrain.bind("<Button-1>", train)
    ctrain.place(x=50,y=150)

    cregister = Canvas(width=160, height=180, highlightthickness=0, background='midnightblue')
    rimage = img_pil.open('face_register.png')
    rkimage = ImageTk.PhotoImage(rimage)
    top.rkimage = rkimage
    cregister.create_image(5, 5, image=rkimage, anchor='nw')
    cregister.create_text(5, 160, text='Register your face', fill='white', font='ABc 11 bold', anchor='nw')
    cregister.bind("<Button-1>", partial(register, id))
    cregister.place(x=300, y=150)

    cattend = Canvas(width=250, height=180, highlightthickness=0, background='midnightblue')
    aimage = img_pil.open('mark_attendance.png')
    akimage = ImageTk.PhotoImage(aimage)
    top.akimage = akimage
    cattend.create_image(50, 5, image=akimage, anchor='nw')
    cattend.create_text(5, 160, text='Get your attendance marked', fill='white', font='ABc 11 bold', anchor='nw')
    cattend.bind("<Button-1>", partial(attend,id))
    cattend.place(x=550, y=150)

    update_attendance(id)



def teacher(id):
    children = top.winfo_children()
    for i in range(2, len(children)):
        children[i].place_forget()

    Label(top, text=f'Faculty ID: {id}', bg='white', fg='black', font=('ABC 20 bold italic')).place(x=620, y=100)
    result = []
    try:
        cursor.execute(f'select name from Users where id={id}')
        result = cursor.fetchall()
        db.commit()
    except Exception as e:
        db.rollback()
        messagebox.showerror('SQL Error', e)
        return

    Label(top, text=f'{result[0][0]}', bg='white', fg='black', font=('ABC 20 bold')).place(x=970, y=100)

    ctrain = Canvas(width=160, height=180, highlightthickness=0, background='midnightblue')
    timage = img_pil.open('train_system1.png')
    tkimage = ImageTk.PhotoImage(timage)
    top.tkimage = tkimage
    ctrain.create_image(5, 5, image=tkimage, anchor='nw')
    ctrain.create_text(5, 160, text='Train your System', fill='white', font='ABc 11 bold', anchor='nw')
    ctrain.bind("<Button-1>", train)
    ctrain.place(x=50, y=150)

    cregister = Canvas(width=160, height=180, highlightthickness=0, background='midnightblue')
    rimage = img_pil.open('face_register.png')
    rkimage = ImageTk.PhotoImage(rimage)
    top.rkimage = rkimage
    cregister.create_image(5, 5, image=rkimage, anchor='nw')
    cregister.create_text(5, 160, text='Register your face', fill='white', font='ABc 11 bold', anchor='nw')
    cregister.bind("<Button-1>", partial(register, id))
    cregister.place(x=300, y=150)

    cattend = Canvas(width=250, height=180, highlightthickness=0, background='midnightblue')
    aimage = img_pil.open('mark_attendance.png')
    akimage = ImageTk.PhotoImage(aimage)
    top.akimage = akimage
    cattend.create_image(50, 5, image=akimage, anchor='nw')
    cattend.create_text(5, 160, text='Get your attendance marked', fill='white', font='ABc 11 bold', anchor='nw')
    cattend.bind("<Button-1>", partial(attend, id))
    cattend.place(x=550, y=150)

    update_attendance(id)

    Label(top, text='Choose Student:', bg='white', fg='brown', font=('ABC 14 bold italic')).place(x=480, y=380)
    records=[]
    try:
        cursor.execute('select name, id from Users where type="S"' )
        records=cursor.fetchall()
        db.commit()
    except Exception as e:
        messagebox.showerror('Sql Exception', e)
        return
    d={}
    for i in records:
        d[i[0]]=i[1]

    C1 = ttk.Combobox(top, values=list(d.keys()), width=10)
    C1.current(0)
    C1.place(x=500, y=430)

    def do():
        curr_name=C1.get()
        curr_id=d[curr_name]
        for_stud(curr_id)
    Button(top, text='View', bg='blue', fg='white', bd=6, font=("ABC 15 bold italic"), command=do).place(x=530, y=480)


def onsignup():
    sn=En.get()
    stype=var.get()
    spwd=Epswd.get()
    scpwd=Ecpswd.get()

    if(sn=='' or stype==0 or spwd=='' or scpwd==''):
        messagebox.showerror('Error','Some Feilds are Empty.\nPlease Check !!')
        return
    if(scpwd!=spwd):
        messagebox.showerror('Error','Password does not match !!')
        return
    if(stype==1):
        stype='S'
    elif(stype==2):
        stype='T'
    else:
        stype='E'

    try:
        cursor.execute('select max(id) from Users')
        mid = cursor.fetchall()
        cid=mid[0][0]
    except Exception as e:
        print(e)


    try:
        cmd=f'insert into Users values({cid+1}, "{sn}", "{stype}", "{spwd}");'
        cursor.execute(cmd)
        db.commit()
    except Exception as e:
        messagebox.showerror('Error ',e)
        db.rollback()
        return

    messagebox.showinfo('Account Added Successfuly','Please note your ID as your login credentials.\nYour ID: %d' %(cid+1))
    Epswd.delete(0, END)
    Ecpswd.delete(0, END)
    En.delete(0, END)
    if stype=='S':
        student(cid+1)
    else:
        teacher(cid+1)




def onlogin():
    sidl=Eid.get()
    spwd=Epwd.get()
    st=varl.get()
    if st==0 or spwd=='' or sidl=='':
        messagebox.showerror('Error', 'Some Feilds are Empty.\nPlease Check !!')
        return

    try:
        sidl=int(sidl)
    except:
        messagebox.showerror('Error', 'Id does not exist.\nPlease Check !!')
        return
    if st==1:
        user_type='S'
    else:
        user_type='T'

    try:
        cursor.execute('select password from Users where id=%d and type="%s"' % (int(sidl), user_type))
        password=cursor.fetchall()
        db.commit()
        if(password==tuple()):
            messagebox.showerror('Error','Please check the Type and ID again !!')
            return
        else:
            if(spwd==password[0][0]):
                Eid.delete(0,END)
                Epwd.delete(0,END)
                if user_type=='S':
                    student(sidl)
                else:
                    teacher(sidl)
            else:
                messagebox.showerror('Error','Incorrect ID or Password !!')
                return
    except Exception as e:
        messagebox.showerror('Error','Following error occured:\n%s' % (e))




Label(top,text='Attendance Automation System', bg='white', fg='indigo', font=('ABC 45 bold italic')).place(x=80,y=10)
Label(top,text='________________________________________________________________________________________________________________________________________________________________________________',bg='white', fg='grey').place(x=35,y=80)

Label(top,text='Create new Account', bg='white', fg='black',font='ABC 20 italic').place(x=170,y=200)
Label(top,text='Already have Account with us?', bg='white', fg='black',font='ABC 20 italic').place(x=720,y=200)
for i in range(135,600,15):
    Label(top,text='|',fg='grey',bg='white').place(x=600,y=i)

var=IntVar()
varl=IntVar()

Label(top,text='Name: ', bg='white', fg='black', font='ABC 15 bold').place(x=70,y=255)
Label(top,text='Type: ', bg='white', fg='black', font='ABC 15 bold').place(x=70,y=320)
Label(top,text='Password : ', bg='white', fg='black', font='ABC 15 bold').place(x=70,y=400)
Label(top,text='Confirm Password: ', bg='white', fg='black', font='ABC 15 bold').place(x=70,y=470)

En=Entry(top,bd=1,bg='white', fg='indigo', font='ABC 12')
En.place(x=300,y=255)

Rs=Radiobutton(top,text='Student', bg='white', fg='indigo', font='ABC 12', variable=var, value=1)
Rs.place(x=270,y=320)
Rt=Radiobutton(top,text='Teacher', bg='white', fg='indigo', font='ABC 12', variable=var, value=2)
Rt.place(x=390,y=320)

Epswd=Entry(top,bd=1,bg='white', fg='indigo', font='ABC 12', show='*')
Epswd.place(x=300,y=400)
Ecpswd=Entry(top,bd=1,bg='white', fg='indigo', font='ABC 12', show='*')
Ecpswd.place(x=300,y=470)

Button(top,text='Sign Up', bg='blue', fg='white', bd=6, font=("ABC 15 bold italic"), command=onsignup).place(x=250,y=550)


Label(top,text='ID: ', bg='white', fg='black', font='ABC 15 bold').place(x=680,y=270)
Label(top,text='Password: ', bg='white', fg='black', font='ABC 15 bold').place(x=680,y=390)
Label(top,text='Type: ', bg='white', fg='black', font='ABC 15 bold').place(x=680,y=330)

Eid=Entry(top,bd=1,bg='white', fg='indigo', font='ABC 12')
Eid.place(x=900,y=270)
Rsl=Radiobutton(top,text='Student', bg='white', fg='indigo', font='ABC 12', variable=varl, value=1)
Rsl.place(x=890,y=330)
Rtl=Radiobutton(top,text='Teacher', bg='white', fg='indigo', font='ABC 12', variable=varl, value=2)
Rtl.place(x=1000,y=330)
Epwd=Entry(top,bd=1,bg='white', fg='indigo', font='ABC 12', show='*')
Epwd.place(x=900,y=390)

Button(top,text='Log In', bg='blue', fg='white', bd=6, font=("ABC 15 bold italic"), command=onlogin).place(x=850,y=450)


top.mainloop()
