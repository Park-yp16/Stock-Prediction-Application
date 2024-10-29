from flask import Flask, render_template, url_for, request, redirect, Response, flash, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from werkzeug.security import generate_password_hash, check_password_hash
import io
import random
import sqlite3 as sql
import database
from model import *
import pytz


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_ERI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = "seniorseminar2021"
db = SQLAlchemy(app)

class User:
    def __init__(username, email, fullname):
        self.username = username
        self.email = email
        self.fullname = fullname

# 157.230.63.172 
@app.route("/", methods=['POST', 'GET'])
def index():
    #Set session to logged_out for the first time visiting the page
    if session.get('user_status') is None:
	    session['user_status'] = 'logged_out'
    if request.method == 'POST':
        stock_info = request.form['content']
    return render_template('index.html')


from datetime import date, datetime

@app.route("/stocks", methods=['POST', 'GET'])
def stocks():
    imagePath = ""
    output=""
    conclusion = ""
    percentage_value = 0
    profit_value = 0
    pred_value = 0
    current = 0

    if request.method == 'POST':
        
        timeSelect = request.form['TimeSelector']
        stockTicker = request.form['StockSelector']
        #SQL Call
        try:
            with sql.connect("database.db") as con:
                cur = con.cursor()

                query = "SELECT modelName FROM models WHERE stock = \"{}\" AND lookup_step = {} ORDER BY updateDate DESC;".format(stockTicker, timeSelect)
               # values = (stockTicker, timeSelect)
                print(query)
                cur.execute(query)
                myResult = cur.fetchone()
                
                myModelName = myResult[0]
                tz = pytz.timezone("US/Eastern")
                date_now = datetime.now(tz).date()
                myStock = MyModel.fromModel(myModelName)

                imagePath = "static/docs/upload/plots/{}_{}days_{}.png".format(stockTicker, timeSelect, date_now)
                pred_value = myStock.getFuturePrice()
                print("output = {}".format(output))
                current = myStock.current
                profit_value = pred_value- current
                percentage_value = float(1-current/pred_value) * 100

                conclusion = (percentage_value >= 0)

                if conclusion:
                    conclusion = "Buy"
                else:
                    conclusion = "Sell"

        except Exception as e:
            print(e)

    return render_template('stocks.html', imagePath = imagePath, output=output, current = round(current,2), pred_value= round(pred_value,2), profit_value = round(profit_value,2), percentage_value= round(percentage_value,4), conclusion= conclusion)

@app.route("/contact", methods=['POST', 'GET'])
def contact():
    if request.method == 'POST':
        stock_info = request.form['content']
    return render_template('contact.html')

@app.route("/about", methods=['POST', 'GET'])
def about():
    if request.method == 'POST':
        stock_info = request.form['content']
    return render_template('about.html')

@app.route("/login", methods=['POST', 'GET'])
def login():
    #Redirect user if already logged in
    if session['user_status'] == 'logged_in':
        return render_template('index.html')

    return render_template('login.html')
    #if request.method == 'POST':
    #stock_info = request.form['content']

@app.route('/loginAttempt', methods=['POST', 'GET'])
def loginAttempt():
    #Redirect user if already logged in
    if session['user_status'] == 'logged_in':
        return render_template('index.html')

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        #Check if username exists
        if not usernameExists(username):
            msg="Username does not exist"
    
            return render_template('login.html', msg = msg)
        
        #Check if passwords match
        if passwordsMatch(username, password):
            msg="Successfully logged in"
            session['user_status'] = 'logged_in'
            session['username'] = username
            return render_template('index.html', msg=msg)
        else:
            msg="Passwords do not match"
            return render_template('login.html', msg = msg)
    else:
        msg = ""
        return render_template("login.html", msg = msg)

@app.route("/signup", methods=['POST', 'GET'])
def signup():
    #Redirect user if already logged in
    if session['user_status'] == 'logged_in':
        return render_template('index.html')

    if request.method == 'POST':
        stock_info = request.form['content']
    return render_template('sign-up.html')

@app.route("/signupAttempt", methods=['POST', 'GET'])
def signupAttempt():
    #Redirect user if already logged in
    msg = ""
    if session['user_status'] == 'logged_in':
        return render_template('index.html')

    if request.method == 'POST':
        if not request.form['name'] or not request.form['id'] or not request.form['email'] or not request.form['password'] or not request.form['password2']:
            msg = "Please fill out all forms before signing up"
            return render_template('sign-up.html', msg = msg)
        
        try:
            name = request.form['name']
            username = request.form['id']
            email = request.form['email']
            password = request.form['password']
            password2 = request.form['password2']

            #Check if passwords match
            if password != password2:
                msg = "Passwords do not match"      
                return render_template('sign-up.html', msg = msg)

            #Check if username already exists
            elif usernameExists(username):
                msg = "Username already exists"       
                return render_template('sign-up.html', msg = msg)

            #Check if email already exists
            elif emailExists(email):
                msg = "Email already exists"
                return render_template('sign-up.html', msg = msg)
            
            #Sign user up
            elif insertUser(name, username, email, password):
                session['user_status'] = 'logged_in'
                session['username'] = username
                msg = "Successfully signed up"
            
            #Something unexpected happened
            else:
                msg = "Error. Please try again"
        except:
            msg = "Something went wrong"
            con.rollback()

        finally:
            return render_template("index.html", msg = msg)
            con.close()

@app.route("/index2", methods=['POST', 'GET'])
def index2():
    if request.method == 'POST':
        stock_info = request.form['content']
    return render_template('index2.html')

@app.route('/list')
def list():
    rows = ""
    con = sql.connect("database.db")
    con.row_factory = sql.Row
    
    cur = con.cursor()
    cur.execute("select * from users")
    
    rows = cur.fetchall(); 
    return render_template("list.html",rows = rows)

@app.route("/mypage", methods=['POST', 'GET'])
def mypage():
    email = ""
    username = ""
    fullname = ""

    con = sql.connect("database.db")
    con.row_factory = sql.Row
    cur = con.cursor()

    #Select user from database
    cur.execute('SELECT * FROM users WHERE username=?', (session['username'],))
    row = cur.fetchone()
    con.close()

    #Get user email and username
    if row is not None:
        email = row['email']
        username = row['username']
        name = row['name']
    else:
        print("Unexpected error. User not found when checking password")

    return render_template('mypage.html', email = email, username = username, name = name)

@app.route("/signout", methods=['POST', 'GET'])
def signout():
    session.pop('username', None)
    session['user_status'] = 'logged_out'
    return render_template('index.html')

@app.route("/updateName", methods=['POST', 'GET'])
def updateName():
    return render_template('updateName.html')

@app.route("/updateNameAttempt", methods=['POST', 'GET'])
def updateNameAttempt():
    email = ""
    username =""
    name = ""
    if request.method == 'POST':
        #Check for any empty forms
        if not request.form['name'] or not request.form['name2']:
            msg = "Please fill out all forms before signing up"
            return render_template('updateName.html', msg = msg)

        try:
            name = request.form['name']
            name2 = request.form['name2']
            
            #Check if names match
            if name != name2:
                msg = "Names do not match"      
                return render_template('updateName.html', msg = msg)

            updateNm(name)

            #Get updated information from database to pass to mypage.html
            msg = "Name successfully changed"
            email = ""
            username = ""
            fullname = ""

            con = sql.connect("database.db")
            con.row_factory = sql.Row
            cur = con.cursor()

            #Select user from database
            cur.execute('SELECT * FROM users WHERE username=?', (session['username'],))
            row = cur.fetchone()
            con.close()

            #Get user email and username
            if row is not None:
                email = row['email']
                username = row['username']
                name = row['name']
            else:
                print("Unexpected error. User not found when checking password")

            return render_template('mypage.html', email = email, username = username, name = name)
        except:
            msg = "Something went wrong"
            con.rollback()

        finally:
            return render_template("mypage.html", email = email, username = username, name = name)
            con.close()

@app.route("/updateEmail", methods=['POST', 'GET'])
def updateEmail():
    return render_template('updateEmail.html')

@app.route("/updateEmailAttempt", methods=['POST', 'GET'])
def updateEmailAttempt():
    email = ""
    name = ""
    username=""
    if request.method == 'POST':
        #Check for any empty forms
        if not request.form['email'] or not request.form['email2']:
            msg = "Please fill out all forms before signing up"
            return render_template('updateEmail.html', msg = msg)

        try:
            email = request.form['email']
            email2 = request.form['email2']
            
            #Check if emails match
            if email != email2:
                msg = "Emails do not match"      
                return render_template('updateEmail.html', msg = msg)
            
            updateEml(email)

            #Get updated information from database to pass to mypage.html
            msg = "Email successfully changed"
            email = ""
            username = ""
            fullname = ""

            con = sql.connect("database.db")
            con.row_factory = sql.Row
            cur = con.cursor()

            #Select user from database
            cur.execute('SELECT * FROM users WHERE username=?', (session['username'],))
            row = cur.fetchone()
            con.close()

            #Get user email and username
            if row is not None:
                email = row['email']
                username = row['username']
                name = row['name']
            else:
                print("Unexpected error. User not found when checking password")

            return render_template('mypage.html', email = email, username = username, name = name)
        except:
            msg = "Something went wrong"
            con.rollback()

        finally:
            return render_template("mypage.html", email = email, username = username, name = name)
            con.close()

@app.route("/updatePassword", methods=['POST', 'GET'])
def updatePassword():
    return render_template('updatePassword.html')

@app.route("/updatePasswordAttempt", methods=['POST', 'GET'])
def updatePasswordAttempt():
    msg = ""
    if request.method == 'POST':
        #Check for any empty forms
        if not request.form['currentPassword'] or not request.form['newPassword']or not request.form['newPassword2']:
            msg = "Please fill out all forms before signing up"
            return render_template('updatePassword.html', msg = msg)
        
        #Check if passwords match
        password = request.form['newPassword']
        password2 = request.form['newPassword2']
        if password != password2:
            msg = "New passwords do not match"
            return render_template('updatePassword.html', msg = msg)

        #Check if passwords match
        username = session['username']
        currentPassword = request.form['currentPassword']
        if passwordsMatch(username, currentPassword):
            try:                
                updatePwd(password)

                #Get updated information from database to pass to mypage.html
                msg = "Password successfully changed"
                email = ""
                username = ""
                fullname = ""

                con = sql.connect("database.db")
                con.row_factory = sql.Row
                cur = con.cursor()

                #Select user from database
                cur.execute('SELECT * FROM users WHERE username=?', (session['username'],))
                row = cur.fetchone()
                con.close()

                #Get user email and username
                if row is not None:
                    email = row['email']
                    username = row['username']
                    name = row['name']
                else:
                    print("Unexpected error. User not found when checking password")

                return render_template('mypage.html', email = email, username = username, name = name)
            except:
                msg = "Something went wrong"
                con.rollback()

            finally:
                return render_template("mypage.html", email = email, username = username, name = name)
                con.close()
        else:
            msg = "Current password is incorrect. Please try again."
            return render_template('updatePassword.html', msg = msg)

@app.route("/updateUsername", methods=['POST', 'GET'])
def updateUsername():
    return render_template('updateUsername.html')

@app.route("/updateUsernameAttempt", methods=['POST', 'GET'])
def updateUsernameAttempt():
    msg = ""
    if request.method == 'POST':
        #Check for any empty forms
        if not request.form['currentPassword'] or not request.form['newUsername']or not request.form['newUsername2']:
            msg = "Please fill out all forms before changing your ID"
            return render_template('updateUsername.html', msg = msg)
        
        #Check if usernames match
        username = request.form['newUsername']
        username2 = request.form['newUsername2']
        if username != username2:
            msg = "New IDs do not match"
            return render_template('updateUsername.html', msg = msg)

        #Check if passwords match
        ID = session['username']
        currentPassword = request.form['currentPassword']
        if passwordsMatch(ID, currentPassword):
            try:                
                updateID(username)

                #Get updated information from database to pass to mypage.html
                msg = "ID successfully changed"
                email = ""
                username = ""
                fullname = ""

                con = sql.connect("database.db")
                con.row_factory = sql.Row
                cur = con.cursor()

                #Select user from database
                cur.execute('SELECT * FROM users WHERE username=?', (session['username'],))
                row = cur.fetchone()
                con.close()

                #Get user email and username
                if row is not None:
                    email = row['email']
                    username = row['username']
                    name = row['name']
                else:
                    print("Unexpected error. User not found when checking ID")

                return render_template('mypage.html', email = email, username = username, name = name)
            except:
                msg = "Something went wrong"
                con.rollback()

            finally:
                return render_template("mypage.html", email = email, username = username, name = name)
                con.close()
        else:
            msg = "Current password is incorrect. Please try again."
            return render_template('updatePassword.html', msg = msg)

def usernameExists(username):
    #Check if username is already in database
    try:
        con = sql.connect("database.db")
        con.row_factory = sql.Row

        cur = con.cursor()
        cur.execute('SELECT * FROM users WHERE username=?', (username,))
        row = cur.fetchone()

        data = 0
        if row is not None:
            if username in row:
                data = 1
        
        #User found in the database
        if data == 1:
            return True
        #User not found in the database
        else:
            return False

    except:
        print("Something went wrong when attempting to find the user in the database")
    finally:
        print("Finished finding user in database")

def emailExists(email):
    #Check if email is already in database
    try:
        con = sql.connect("database.db")
        con.row_factory = sql.Row

        cur = con.cursor()
        cur.execute('SELECT * FROM users WHERE email=?', (email,))
        row = cur.fetchone()

        data = 0
        if row is not None:
            if email in row:
                data = 1
        
        #Email found in the database
        if data == 1:
            return True
        #Email not found in the database
        else:
            return False

    except:
        print("Something went wrong when attempting to find the email in the database")
    finally:
        print("Finished finding email in database")

def insertUser(name, username, email, password):
#Insert new user into table
    try:
        with sql.connect("database.db") as con:
            cur = con.cursor()

            #hash password using sha256
            hashedPassword = generate_password_hash(password, method='sha256')

            #insert user into table
            cur.execute("INSERT INTO users (name, username, email, password) VALUES (?, ?, ?, ?)", (name, username, email, hashedPassword))
            con.commit()
    except:
        print("Something went wrong attempting to insert user into database")
    finally:
        print("Successfully inserted user into database")
        return True;

def passwordsMatch(username, password):
    try:
        con = sql.connect("database.db")
        con.row_factory = sql.Row
        cur = con.cursor()

        #Select user from database
        cur.execute('SELECT * FROM users WHERE username=?', (username,))
        row = cur.fetchone()

        #Check if password matches password in database
        if row is not None:
            if username in row:
                return check_password_hash(row['password'], password)
            else:
                print("Unexpected error occured. User not found when checking password")
        else:
            print("Unexpected error. User not found when checking password")

    except:
        print("Something went wrong when authenticating the user")
    finally:
        con.close
        print("End of password match function")

def updateNm(name):
    try:
        with sql.connect("database.db") as con:
            cur = con.cursor()
            cur.execute('UPDATE users set name =? WHERE username=?', (name,session['username']))
            con.commit()
    except:
        print("Something went wrong attempting to change user's name in database")
    finally:
        print("Successfully changed user's name in database")
        return True;

def updateEml(email):
    try:
        with sql.connect("database.db") as con:
            cur = con.cursor()
            cur.execute('UPDATE users set email =? WHERE username=?', (email,session['username']))
            con.commit()
    except:
        print("Something went wrong attempting to change user's name in database")
    finally:
        print("Successfully changed user's email in database")
        return True;

def updatePwd(password):
    try:
        with sql.connect("database.db") as con:
            cur = con.cursor()

            #hash password using sha256
            hashedPassword = generate_password_hash(password, method='sha256')

            cur.execute('UPDATE users set password =? WHERE username=?', (hashedPassword,session['username']))
            con.commit()
    except:
        print("Something went wrong attempting to change user's password in database")
    finally:
        print("Successfully changed user's password in database")
        return True;

def updateID(username):
    try:
        with sql.connect("database.db") as con:
            cur = con.cursor()

            cur.execute('UPDATE users set username =? WHERE username=?', (username,session['username']))
            con.commit()

        session.pop('username', None)
        session['username'] = username
            
    except:
        print("Something went wrong attempting to change user's password in database")
    finally:
        print("Successfully changed user's password in database")
        return True;

if __name__ == "__main__":
    app.run(debug=True)
