import os
import json
import io
import re
import shutil
from datetime import datetime
from collections import Counter
import smtplib
from email.message import EmailMessage

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    abort,
    jsonify,
    send_file,
)
from werkzeug.utils import secure_filename

# === AUTHENTICATION IMPORTS ===
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from auth import db, bcrypt, User

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Preformatted,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import markdown

from format_con import convert_file, RAW_DIR, OUT_DIR
from rag_engine import generate_summary, query_with_rag
from live_ingest import index_normalized_file

# Import ChromaDB client for session management
from chromadb import PersistentClient

ALLOWED_EXTENSIONS = {".xml", ".json", ".csv", ".log", ".txt"}
VECTOR_DB_PATH = "./DefenSight AI_db"

app = Flask(__name__)
app.secret_key = "super-secret-change-me"

# === DATABASE CONFIGURATION ===
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# === INITIALIZE AUTHENTICATION ===
db.init_app(app)
bcrypt.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access DefenSight AI'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# === CREATE DATABASE AND DEFAULT USER ===
with app.app_context():
    db.create_all()
    if User.query.count() == 0:
        admin = User(username='admin', email='admin@defensight.ai')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("✅ Default user created: admin / admin123")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ============== AUTHENTICATION ROUTES ==============

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('upload_form'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=request.form.get('remember', False))
            user.update_last_login()
            flash(f'Welcome, {user.username}!', 'success')
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('upload_form'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('upload_form'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        # Validation
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
        else:
            # Create user
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please log in', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    """Logout"""
    username = current_user.username
    logout_user()
    flash(f'Goodbye, {username}!', 'info')
    return redirect(url_for('login'))


# ============== UTILITY FUNCTIONS ==============

def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def get_db_stats():
    """Get current database statistics"""
    try:
        client = PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_or_create_collection("defensight_ai")
        return {
            "total_documents": collection.count(),
            "has_data": collection.count() > 0
        }
    except Exception as e:
        print(f"Error getting DB stats: {e}")
        return {"total_documents": 0, "has_data": False}


def clear_session():
    """Clear all data from the current session"""
    try:
        # Clear ChromaDB
        client = PersistentClient(path=VECTOR_DB_PATH)
        try:
            client.delete_collection("defensight_ai")
        except:
            pass
        client.get_or_create_collection("defensight_ai")
        
        # Clear normalized files
        if os.path.exists(OUT_DIR):
            for file in os.listdir(OUT_DIR):
                file_path = os.path.join(OUT_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        # Clear raw files
        if os.path.exists(RAW_DIR):
            for file in os.listdir(RAW_DIR):
                file_path = os.path.join(RAW_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        return True, "Session cleared successfully"
    except Exception as e:
        return False, f"Error clearing session: {str(e)}"


def build_normalization_summary(norm_path, max_rows=100):
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalized file not found: {norm_path}")

    with open(norm_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    total_records = len(data)

    if total_records == 0:
        summary = {
            "file_name": os.path.basename(norm_path),
            "total_records": 0,
            "type_counts": {},
            "key_fields": [],
        }
        return summary, [], []

    type_counts = Counter(entry.get("type", "unknown") for entry in data)

    key_counter = Counter()
    for entry in data[:200]:
        if isinstance(entry, dict):
            key_counter.update(entry.keys())

    preferred = [
        "timestamp",
        "description",
        "type",
        "source_file",
        "srcip",
        "dstip",
        "proto",
        "severity",
        "attack_cat",
        "line_number",
    ]

    columns = []
    for k in preferred:
        if k in key_counter and k not in columns:
            columns.append(k)

    for k, _ in key_counter.most_common():
        if k not in columns:
            columns.append(k)
        if len(columns) >= 8:
            break

    rows = []
    if max_rows > 0:
        for entry in data[:max_rows]:
            row = {col: entry.get(col, "") for col in columns}
            rows.append(row)

    summary = {
        "file_name": os.path.basename(norm_path),
        "total_records": total_records,
        "type_counts": dict(type_counts),
        "key_fields": columns,
    }

    return summary, columns, rows


# ============== PROTECTED ROUTES ==============

@app.route("/")
@login_required
def upload_form():
    db_stats = get_db_stats()
    return render_template("upload.html", db_stats=db_stats, user=current_user)


@app.route("/session/stats", methods=["GET"])
@login_required
def session_stats():
    """API endpoint to get current session stats"""
    stats = get_db_stats()
    return jsonify(stats)


@app.route("/session/clear", methods=["POST"])
@login_required
def clear_session_route():
    """API endpoint to clear the current session"""
    success, message = clear_session()
    return jsonify({
        "success": success,
        "message": message,
        "stats": get_db_stats()
    })


@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    """
    Upload one or more log files, normalize them, and index into ChromaDB.
    Now supports session mode selection.
    """
    # Check session mode
    session_mode = request.form.get("session_mode", "append")
    
    files = []

    if "logfiles" in request.files:
        files = request.files.getlist("logfiles")

    if not files and "logfile" in request.files:
        single = request.files["logfile"]
        if single.filename:
            files = [single]

    if not files:
        flash("No files uploaded.", "warning")
        return redirect(url_for("upload_form"))

    valid_files = []
    invalid_files = []

    for file in files:
        if file.filename == "":
            invalid_files.append("(empty filename)")
            continue

        if not allowed_file(file.filename):
            invalid_files.append(file.filename)
        else:
            valid_files.append(file)

    if not valid_files:
        flash(
            "No valid files. Unsupported or empty filenames. "
            "Allowed: XML, JSON, CSV, LOG, TXT.",
            "error"
        )
        return redirect(url_for("upload_form"))

    if invalid_files:
        flash(
            f"These files were skipped (unsupported type or empty): "
            f"{', '.join(invalid_files)}",
            "warning"
        )

    # Clear session if requested
    if session_mode == "new":
        success, message = clear_session()
        if success:
            flash("✅ Started new session - previous data cleared", "success")
        else:
            flash(f"⚠️ Warning: {message}", "warning")

    # Process + index each valid file
    indexed_count = 0
    for file in valid_files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(RAW_DIR, filename)
        file.save(save_path)

        # Normalize (creates JSON in OUT_DIR)
        convert_file(save_path)

        # Index into ChromaDB
        normalized_filename = filename.rsplit(".", 1)[0] + ".json"
        normalized_path = os.path.join(OUT_DIR, normalized_filename)

        try:
            index_normalized_file(normalized_path)
            indexed_count += 1
            print(f"Indexed: {normalized_filename}")
        except Exception as e:
            print(f"Failed to index {normalized_filename}: {e}")
            flash(f"Warning: File {filename} uploaded but indexing failed.", "warning")

    if session_mode == "new":
        flash(f"New session started with {indexed_count} file(s)!", "success")
    else:
        flash(f"Successfully added {indexed_count} file(s) to existing session!", "success")
    
    return redirect(url_for("list_normalized_files"))


@app.route("/normalize/<fname>")
@login_required
def normalization_view(fname):
    norm_path = os.path.join(OUT_DIR, fname)

    if not os.path.exists(norm_path):
        abort(404, description=f"Normalized file not found: {fname}")

    summary, columns, rows = build_normalization_summary(norm_path)

    return render_template(
        "normalize.html",
        summary=summary,
        columns=columns,
        rows=rows,
        user=current_user,
    )


@app.route("/normalized_files")
@login_required
def list_normalized_files():
    summaries = []

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    for fname in sorted(os.listdir(OUT_DIR)):
        if not fname.lower().endswith(".json"):
            continue

        norm_path = os.path.join(OUT_DIR, fname)
        try:
            summary, _, _ = build_normalization_summary(norm_path, max_rows=0)
            summaries.append(summary)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    db_stats = get_db_stats()
    return render_template("normalized_list.html", summaries=summaries, db_stats=db_stats, user=current_user)


@app.route("/analysis")
@login_required
def analysis_view():
    tech_md = generate_summary("technical")
    exec_md = generate_summary("executive")

    tech_html = markdown.markdown(tech_md, extensions=["fenced_code", "tables"])
    exec_html = markdown.markdown(exec_md, extensions=["fenced_code", "tables"])

    return render_template(
        "index.html",
        technical=tech_html,
        executive=exec_html,
        user=current_user,
    )


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json() or {}
    user_msg = (data.get("message") or "").strip()

    if not user_msg:
        return jsonify({"reply": "⚠️ Please provide a valid message."}), 400

    try:
        reply_md = query_with_rag(user_msg)
        reply_html = markdown.markdown(reply_md, extensions=["fenced_code", "tables"])
        return jsonify({"reply": reply_html})
    except Exception as e:
        return jsonify({"reply": f"❌ Error during chat: {str(e)}"}), 500


# === PDF REPORT GENERATION (shared helper) ===
def build_report_pdf_bytes() -> bytes:
    """Generate the DefenSight AI PDF report and return it as raw bytes."""

    tech_md = generate_summary("technical")
    exec_md = generate_summary("executive")

    buffer = io.BytesIO()

    # --- Create PDF doc with margins ---
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=70,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    # --- Custom styles ---
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#0B3D91"),
        spaceAfter=18,
        alignment=1,  # center
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#333333"),
        backColor=colors.HexColor("#E8F0FE"),
        leftIndent=0,
        spaceBefore=12,
        spaceAfter=12,
        borderPadding=(4, 6, 4),
    )

    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.HexColor("#0B3D91"),
        spaceBefore=10,
        spaceAfter=6,
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#16355B"),
        spaceBefore=8,
        spaceAfter=4,
    )

    h3_style = ParagraphStyle(
        "H3",
        parent=styles["Heading3"] if "Heading3" in styles else styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#1F2933"),
        spaceBefore=6,
        spaceAfter=3,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=18,
        bulletIndent=9,
    )

    num_style = ParagraphStyle(
        "Numbered",
        parent=body_style,
        leftIndent=18,
        bulletIndent=9,
    )

    code_style = ParagraphStyle(
        "Code",
        parent=styles["Code"] if "Code" in styles else styles["BodyText"],
        fontName="Courier",
        fontSize=9,
        leading=11,
        backColor=colors.HexColor("#2E2E2E"),
        textColor=colors.white,
        leftIndent=6,
        rightIndent=6,
        spaceBefore=6,
        spaceAfter=6,
    )

    # --- Inline markdown -> ReportLab helper ---
    def inline_md(text: str) -> str:
        """
        Convert simple inline markdown (**bold**, *italic*, `code`)
        into ReportLab's mini-HTML (<b>, <i>, <font>).
        """
        # escape &, <, >
        text = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )

        # bold: **text**
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

        # inline code: `text`
        text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)

        # italic: *text*  (avoid **bold** which is already handled)
        text = re.sub(
            r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
            r"<i>\1</i>",
            text,
        )

        return text

    story = []

    # --- Header/footer callback ---
    def add_header_footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setTitle("DefenSight AI – Security Analysis Report")
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.grey)

        # Header
        canvas.drawString(
            doc.leftMargin,
            A4[1] - 40,
            "DefenSight AI – Security Analysis Report",
        )

        # Footer
        canvas.drawRightString(
            A4[0] - doc.rightMargin,
            25,
            f"Page {doc_obj.page}",
        )
        canvas.restoreState()

    # --- Markdown → Flowables helper ---
    def add_md_section(md_text: str, section_title: str):
        story.append(Paragraph(section_title, section_style))
        story.append(Spacer(1, 12))

        lines = md_text.splitlines()
        in_code_block = False
        code_lines = []

        for raw_line in lines:
            # Handle fenced code blocks ```
            if raw_line.strip().startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    code_lines = []
                else:
                    # closing fence → flush code block
                    if code_lines:
                        code_text = "\n".join(code_lines)
                        story.append(Preformatted(code_text, code_style))
                        story.append(Spacer(1, 6))
                    in_code_block = False
                continue

            if in_code_block:
                code_lines.append(raw_line.rstrip("\n"))
                continue

            line = raw_line.strip()

            if not line:
                story.append(Spacer(1, 4))
                continue

            # Headings
            if line.startswith("### "):
                story.append(Paragraph(inline_md(line[4:].strip()), h3_style))
            elif line.startswith("## "):
                story.append(Paragraph(inline_md(line[3:].strip()), h2_style))
            elif line.startswith("# "):
                story.append(Paragraph(inline_md(line[2:].strip()), h1_style))

            # Bullet list (- or *)
            elif line.startswith("- ") or line.startswith("* "):
                bullet_text = line[2:].strip()
                story.append(
                    Paragraph(f"• {inline_md(bullet_text)}", bullet_style)
                )

            # Numbered list (1. item)
            elif re.match(r"^\d+\.\s+", line):
                bullet_text = re.sub(r"^\d+\.\s+", "", line)
                story.append(
                    Paragraph(inline_md(bullet_text), num_style)
                )

            else:
                # Normal paragraph
                story.append(Paragraph(inline_md(line), body_style))

        # If file ended inside an open code block (no closing ```), flush it
        if in_code_block and code_lines:
            code_text = "\n".join(code_lines)
            story.append(Preformatted(code_text, code_style))

        story.append(Spacer(1, 14))

    # --- Cover / main title ---
    story.append(Paragraph("DefenSight AI – Security Analysis Report", title_style))
    story.append(
        Paragraph(
            datetime.now().strftime("Generated on %Y-%m-%d %H:%M:%S"),
            body_style,
        )
    )
    story.append(Spacer(1, 24))

    # --- Technical Summary ---
    add_md_section(tech_md, "Technical Summary")

    # Page break between sections
    story.append(PageBreak())

    # --- Executive Summary ---
    add_md_section(exec_md, "Executive Summary")

    # --- Build PDF ---
    doc.build(
        story,
        onFirstPage=add_header_footer,
        onLaterPages=add_header_footer,
    )

    buffer.seek(0)
    return buffer.getvalue()


@app.route("/download_report")
@login_required
def download_report():
    """Download the generated DefenSight AI report as a PDF."""
    pdf_bytes = build_report_pdf_bytes()
    return send_file(
        io.BytesIO(pdf_bytes),
        as_attachment=True,
        download_name="DefenSight_AI_Report.pdf",
        mimetype="application/pdf",
    )


@app.route("/email_report", methods=["POST"])
@login_required
def email_report():
    """
    Send the generated DefenSight AI report PDF via email.

    Expects JSON body:
        {
            "to": "recipient@example.com",
            "subject": "optional subject",
            "body": "optional plain text body"
        }

    Uses Gmail SMTP with credentials from environment:
        GMAIL_USER, GMAIL_APP_PASSWORD
    """
    data = request.get_json() or {}
    to_email = (data.get("to") or "").strip()
    subject = data.get("subject") or "DefenSight AI - Security Analysis Report"
    body_text = data.get("body") or (
        "Please find attached the DefenSight AI security analysis report."
    )

    if not to_email:
        return jsonify({"ok": False, "error": "Missing 'to' email address"}), 400

    gmail_user = os.getenv("GMAIL_USER")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")

    if not gmail_user or not gmail_app_password:
        return jsonify({
            "ok": False,
            "error": "Email credentials not configured. Set GMAIL_USER and GMAIL_APP_PASSWORD environment variables.",
        }), 500

    # Generate the PDF bytes
    pdf_bytes = build_report_pdf_bytes()

    # Build the email
    msg = EmailMessage()
    msg["From"] = gmail_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body_text)

    msg.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename="DefenSight_AI_Report.pdf",
    )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_user, gmail_app_password)
            smtp.send_message(msg)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "message": f"Report emailed to {to_email}."})


if __name__ == "__main__":
    app.run(debug=True)