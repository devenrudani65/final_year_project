import smtplib
from email.message import EmailMessage

def send_email_report(receiver_email, pdf_file):

    sender_email = "devenrudani65@gmail.com"
    app_password = "eacqeurwszxigvyw"

    msg = EmailMessage()

    msg["Subject"] = "Your CBC Health Report"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    msg.set_content(
    "Your AI generated CBC health report is attached."
    )

    msg.add_attachment(
        pdf_file.getvalue(),
        maintype="application",
        subtype="pdf",
        filename="cbc_report.pdf"
    )

    with smtplib.SMTP_SSL("smtp.gmail.com",465) as smtp:

        smtp.login(sender_email,app_password)

        smtp.send_message(msg)