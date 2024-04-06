# מערכת לבחירת בגדים בארון הביתי

## גרסה: 1.0
מערכת משולבת לניהול ארון בגדים המאפשרת:
- להוסיף תמונות של בגדים מהארון שלך
- לזהות באופן אוטומטי את סוג הבגד בתמונה (למשל, חולצה, מכנסיים, נעליים)
- לסווג את הבגדים לפי גוף (חלק עליון, חלק תחתון, שמלה), סוג (חולצה, מכנסיים, סנדלים וכו') ומזג אוויר (חם, קר, שניהם)
- ליצור טבלה מסודרת של כל הבגדים בארון שלך
- למחוק פריטים מהארון שלך
- לקבל המלצות לבוש בהתבסס על מזג האוויר, אירוע, שעה ומיקום

  
### התקנה

1. וודא שיש לך Python 3 מותקן במחשב שלך.
2. התקן את כל הדרישות על ידי הרצת הפקודות הבאות:

```bash
pip install -r requirements.txt
```

3. שכפל את הקוד למחשב שלך.


### שימוש

הפעל את הקוד באמצעות הפקודה
```bash
streamlit run main.py
```

**בחר באחת מהפעולות הבאות:**

- **הוספת פריט חדש:** לחץ על הכפתור "הוספת פריט חדש" כדי להעלות תמונה של בגד מהארון שלך. היישום יזהה את סוג הבגד באופן אוטומטי ויסווג אותו.
- **מחיקת פריט:** לחץ על הכפתור "הסרת פריט" כדי למחוק פריט מהארון שלך.
- **יצירת סט המלצה:** לחץ על הכפתור "צור סט" כדי לקבל המלצות לבוש בהתבסס על מזג האוויר, אירוע, שעה ומיקום.
- **הצגת מסד נתוני בגדים:** לחץ על הכפתור "מסד נתוני בגדים" כדי לראות טבלה של כל הבגדים בארון שלך.

- 

### פיתוח 

יישום זה מיועד למטרות הדגמה בלבד. הוא ניתן להתאמה אישית ולהרחבה בהתאם לצרכים שלך.



### תודות

יישום זה נעזר בחבילות קוד פתוח רבות, כולל:

- [Streamlit](https://docs.streamlit.io/)
- [Keras](https://keras.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)












