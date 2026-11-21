"""Build a compact IEEE-style project report for RoadGuard."""
from pathlib import Path
import csv

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, FrameBreak, NextPageTemplate, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image, Flowable,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "RoadGuard_IEEE_Report.pdf"
PAGE_W, PAGE_H = letter
MARGIN = 0.55 * inch
GAP = 0.22 * inch
COL_W = (PAGE_W - 2 * MARGIN - GAP) / 2

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=22, alignment=TA_CENTER, spaceAfter=5))
styles.add(ParagraphStyle(name="PaperAuthor", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=7))
styles.add(ParagraphStyle(name="LinkBanner", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#1d5838"), spaceBefore=2, spaceAfter=7))
styles.add(ParagraphStyle(name="Abstract", parent=styles["Normal"], fontName="Helvetica", fontSize=8.8, leading=11, alignment=TA_JUSTIFY, spaceAfter=3))
styles.add(ParagraphStyle(name="Keywords", parent=styles["Normal"], fontName="Helvetica", fontSize=8.4, leading=10.4, alignment=TA_LEFT, spaceAfter=1))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=12, alignment=TA_CENTER, spaceBefore=5, spaceAfter=3, keepWithNext=True))
styles.add(ParagraphStyle(name="Subsection", parent=styles["Heading3"], fontName="Helvetica-BoldOblique", fontSize=8.5, leading=10, alignment=TA_LEFT, spaceBefore=4, spaceAfter=2, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyIEEE", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9, leading=11.3, alignment=TA_JUSTIFY, spaceAfter=3.5))
styles.add(ParagraphStyle(name="SmallIEEE", parent=styles["BodyText"], fontName="Times-Roman", fontSize=8, leading=9.8, alignment=TA_JUSTIFY, spaceAfter=2.5))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontName="Helvetica", fontSize=7.5, leading=9, alignment=TA_CENTER, spaceBefore=2, spaceAfter=4))
styles.add(ParagraphStyle(name="Ref", parent=styles["BodyText"], fontName="Times-Roman", fontSize=7.5, leading=9.2, leftIndent=11, firstLineIndent=-11, spaceAfter=2))


def P(text, style="BodyIEEE"):
    return Paragraph(text, styles[style])


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(MARGIN, PAGE_H - 0.32 * inch, "CSE471 | ROADGUARD PROJECT REPORT")
    canvas.drawRightString(PAGE_W - MARGIN, 0.3 * inch, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#aaaaaa"))
    canvas.setLineWidth(0.35)
    canvas.line(MARGIN, PAGE_H - 0.38 * inch, PAGE_W - MARGIN, PAGE_H - 0.38 * inch)
    canvas.restoreState()


class PipelineFigure(Table):
    def __init__(self):
        labels = ["Road image<br/>JPG / PNG / WebP", "YOLO detector<br/>or CV fallback", "Boxes + class<br/>confidence", "PSI heuristic<br/>and triage", "Web / dashboard<br/>history + map"]
        cells = []
        for label in labels:
            cells.append(Paragraph(label, ParagraphStyle("box", fontName="Helvetica", fontSize=6.8, leading=8, alignment=TA_CENTER, textColor=colors.HexColor("#173a2a"))))
        super().__init__([cells], colWidths=[COL_W / 5] * 5, rowHeights=[0.47 * inch])
        self.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edf4ef")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#557c63")),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9db6a3")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]))


class ArchitectureDiagram(Flowable):
    """Small vector diagram showing the implemented application components."""
    def __init__(self, width=COL_W, height=2.03*inch):
        super().__init__()
        self.width, self.height = width, height

    def draw(self):
        c = self.canv
        green = colors.HexColor("#24583b")
        pale = colors.HexColor("#edf4ef")
        blue = colors.HexColor("#eef3f8")
        gray = colors.HexColor("#58636b")
        boxes = [
            (2, 89, 52, 40, "Browser UI", "Upload / history"),
            (63, 89, 53, 40, "FastAPI", "Validation / API"),
            (126, 105, 49, 40, "YOLO", "Model inference"),
            (126, 54, 49, 40, "CV fallback", "Morphology"),
            (185, 89, 53, 40, "Severity", "PSI heuristic"),
            (63, 15, 53, 30, "SQLite", "Accounts"),
            (185, 15, 53, 30, "CSV + browser", "Telemetry / scans"),
        ]
        for x, y, w, h, title, sub in boxes:
            c.setFillColor(pale if title in ("FastAPI", "Severity") else blue)
            c.setStrokeColor(green if title in ("FastAPI", "Severity") else colors.HexColor("#9aaab2"))
            c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
            c.setFillColor(green if title in ("FastAPI", "Severity") else colors.HexColor("#25323a"))
            c.setFont("Helvetica-Bold", 7.2)
            c.drawCentredString(x+w/2, y+h-14, title)
            c.setFillColor(gray); c.setFont("Helvetica", 5.7)
            c.drawCentredString(x+w/2, y+8, sub)
        def arrow(x1, y1, x2, y2):
            c.setStrokeColor(gray); c.setFillColor(gray); c.setLineWidth(.8)
            c.line(x1, y1, x2, y2)
            import math
            a = math.atan2(y2-y1, x2-x1)
            for da in (2.55, -2.55):
                c.line(x2, y2, x2+4*math.cos(a+da), y2+4*math.sin(a+da))
        arrow(54,109,63,109); arrow(116,111,126,124); arrow(116,104,126,74)
        arrow(175,124,185,111); arrow(175,74,185,101)
        arrow(89,89,89,45); arrow(211,89,211,45)


class SeverityFlow(Flowable):
    def __init__(self, width=COL_W, height=.82*inch):
        super().__init__(); self.width=width; self.height=height
    def draw(self):
        c=self.canv
        labels=["bbox pixels", "fixed scale", "area + depth", "PSI rules", "P1 / P2 / P3"]
        w=(self.width-28)/5
        for i,label in enumerate(labels):
            x=i*(w+7)
            c.setFillColor(colors.HexColor("#fff5df") if i==4 else colors.HexColor("#f2f5f3"))
            c.setStrokeColor(colors.HexColor("#b9c8be")); c.roundRect(x,13,w,31,4,fill=1,stroke=1)
            c.setFillColor(colors.HexColor("#27352d")); c.setFont("Helvetica",6.4); c.drawCentredString(x+w/2,25,label)
            if i<4:
                c.setStrokeColor(colors.HexColor("#6b7b70")); c.line(x+w,28,x+w+5,28)
                c.line(x+w+5,28,x+w+2,30); c.line(x+w+5,28,x+w+2,26)


class PriorityChart(Flowable):
    """Descriptive priority counts from the bundled telemetry CSV."""
    def __init__(self, width=PAGE_W-2*MARGIN, height=1.55*inch):
        super().__init__(); self.width=width; self.height=height
        with (ROOT/'Data'/'pothole_telemetry_records.csv').open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        self.n=len(rows)
        labels={"P1 Critical":lambda v:v.startswith("P1"),"P2 Moderate":lambda v:v.startswith("P2"),"P3 Monitor":lambda v:v.startswith("P3"),"Normal":lambda v:"Normal" in v}
        self.counts={label:sum(test(row.get("hazard_priority", "")) for row in rows) for label,test in labels.items()}
    def draw(self):
        c=self.canv
        palette={"P1 Critical":"#b55252","P2 Moderate":"#c48b22","P3 Monitor":"#43815b","Normal":"#71808a"}
        c.setFillColor(colors.HexColor("#26352d")); c.setFont("Helvetica-Bold",8)
        c.drawString(0,self.height-12,f"Priority labels in bundled telemetry sample (n={self.n})")
        max_count=max(self.counts.values()) or 1
        label_w=86; bar_x=label_w; bar_max=self.width-label_w-30
        for i,(label,count) in enumerate(self.counts.items()):
            y=self.height-32-i*18
            c.setFillColor(colors.HexColor("#48544d")); c.setFont("Helvetica",7); c.drawString(0,y+3,label)
            c.setFillColor(colors.HexColor("#edf0ee")); c.roundRect(bar_x,y,bar_max,10,3,fill=1,stroke=0)
            c.setFillColor(colors.HexColor(palette[label])); c.roundRect(bar_x,y,max(3,bar_max*count/max_count),10,3,fill=1,stroke=0)
            c.setFillColor(colors.HexColor("#26352d")); c.drawString(bar_x+bar_max+6,y+2,str(count))


def build():
    doc = BaseDocTemplate(str(OUTPUT), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=0.53*inch, bottomMargin=0.48*inch, title="RoadGuard: Image-Based Road Defect Inspection", author="CSE471 Project")
    title_bottom = PAGE_H - 3.30*inch
    title_frame = Frame(MARGIN, title_bottom, PAGE_W - 2*MARGIN, 2.75*inch, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="title")
    first_col_height = title_bottom - (MARGIN + 0.05*inch)
    left = Frame(MARGIN, MARGIN + 0.05*inch, COL_W, first_col_height, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="left")
    right = Frame(MARGIN + COL_W + GAP, MARGIN + 0.05*inch, COL_W, first_col_height, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="right")
    first = PageTemplate(id="First", frames=[title_frame, left, right], onPage=header_footer)
    two_left = Frame(MARGIN, MARGIN + 0.05*inch, COL_W, PAGE_H - 0.97*inch, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="left2")
    two_right = Frame(MARGIN + COL_W + GAP, MARGIN + 0.05*inch, COL_W, PAGE_H - 0.97*inch, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="right2")
    gallery_frame = Frame(MARGIN, MARGIN + 0.05*inch, PAGE_W-2*MARGIN, PAGE_H-0.97*inch, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="gallery")
    doc.addPageTemplates([first, PageTemplate(id="TwoCol", frames=[two_left, two_right], onPage=header_footer), PageTemplate(id="Gallery", frames=[gallery_frame], onPage=header_footer)])

    story = [
        P("RoadGuard: An Image-Based Road Defect Inspection and Triage System", "PaperTitle"),
        P("Ankit Kesharwani &nbsp;&nbsp;|&nbsp;&nbsp; Devanshbu Kumar<br/>CSE471 Project Report &nbsp; | &nbsp; Computer Vision and Deep Learning", "PaperAuthor"),
        Table([[P("GITHUB REPOSITORY<br/><link href='https://github.com/Devanshu-Dhyanu/Pothole-detection-systemm' color='#1d5838'>github.com/Devanshu-Dhyanu/<br/>Pothole-detection-systemm</link>","LinkBanner"),P("LIVE DEPLOYMENT<br/><link href='https://pothole-detection-systemm.onrender.com' color='#1d5838'>pothole-detection-systemm.onrender.com</link>","LinkBanner")]], colWidths=[(PAGE_W-2*MARGIN)/2]*2, style=TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#edf4ef")),("BOX",(0,0),(-1,-1),.7,colors.HexColor("#9db6a3")),("INNERGRID",(0,0),(-1,-1),.6,colors.HexColor("#c5d5ca")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),6)])),
        P("<b>Abstract—</b> Road surface defects such as potholes and cracks create safety risks and require timely maintenance. This report presents RoadGuard, a prototype that accepts road images, runs object detection, estimates defect severity, and presents an inspection result through browser and Streamlit interfaces. The system combines an Ultralytics YOLO inference path with a classical image-processing fallback, a rule-based severity engine, a FastAPI service, local scan history, and a telemetry view. The repository contains pretrained model artifacts and a 50-row telemetry CSV. Its current depth, area, priority, and repair-cost outputs are heuristic estimates: image-space boxes are converted using fixed scale assumptions, without per-camera calibration or measured depth ground truth. The repository also contains benchmark metric files, but the included evaluation script writes fixed values rather than computing metrics from predictions and annotations. Accordingly, no detector accuracy or real-world engineering performance is claimed here. The report documents the implemented pipeline, its interfaces, limits, and a reproducible validation plan.", "Abstract"),
        P("<b>Index Terms—</b> road damage detection, pothole, YOLO, computer vision, severity estimation, inspection system.", "Keywords"),
        NextPageTemplate("TwoCol"),
        P("I. INTRODUCTION", "Section"),
        P("Road inspection is often performed through visual review and periodic surveys. Image-based detection can help organize observations by highlighting candidate defects and making inspection records easier to review. RoadGuard is a course-project prototype for this workflow. It is designed to analyze one uploaded road image at a time, annotate detections, assign a rule-based priority, and expose the result through a web interface and API."),
        P("The contribution is an end-to-end integration rather than a newly validated detector: (1) image upload and validation; (2) model-based detection with a classical fallback; (3) heuristic severity and repair triage; and (4) a browser workflow with local scan history, map visualization, and report export. The system is useful as a demonstration scaffold, while its outputs require calibration and field validation before operational use."),
        P("II. SYSTEM OVERVIEW", "Section"),
        P("Fig. 1 summarizes the processing path. The browser sends an image to FastAPI, which verifies image bytes and format, invokes the detector, computes a severity report, and returns detections plus an annotated JPEG. The browser can save scan summaries in IndexedDB and optionally attach geolocation. A separate Streamlit dashboard offers image inspection, telemetry mapping, repair estimates, and model information."),
        PipelineFigure(), P("Fig. 1. RoadGuard image inspection and reporting pipeline.", "Caption"),
        P("A. Image input and API", "Subsection"),
        P("The detection endpoint accepts JPEG, PNG, and WebP files up to 10 MB. It checks the decoded image format and readability before inference. Confidence and IoU thresholds are query parameters. The service returns a detection list, elapsed time, severity fields, and a base64-encoded annotation. A health endpoint reports service availability and whether a model is loaded."),
        ArchitectureDiagram(), P("Fig. 2. Application architecture and data paths implemented in the repository.", "Caption"),
        P("B. Detection path", "Subsection"),
        P("RoadGuardDetector searches configured weight paths and loads the first available Ultralytics YOLO artifact. YOLO frames detection as a single-stage prediction problem [1], while Ultralytics supplies the inference interface used here [4]. The repository includes best.pt, ONNX exports, and generic YOLO weights. If Ultralytics or usable weights are unavailable, a fallback analyzes dark regions with thresholding, morphology, contours, and image gradients. The fallback is an image heuristic and should not be interpreted as learned defect recognition. Class labels and class-color mappings are defined in the detector implementation."),
        P("C. User interface and records", "Subsection"),
        P("The browser application supports account creation and sign-in, optional Google OAuth, image review, location tagging, scan history, CSV export, printable reports, and a map. Browser scan history is stored locally rather than synchronized to an account. The Streamlit dashboard is a separate interface. The supplied telemetry CSV has 50 records and 17 fields, including coordinates, defect counts, environmental descriptors, recommended action, and cost. Its data dictionary describes the fields; the records should be treated as bundled project data unless independently sourced and audited."),
        P("III. SEVERITY ESTIMATION", "Section"),
        P("The severity engine converts each pothole bounding box into approximate physical dimensions using a fixed lower-image scale. Let w and h be box width and height in pixels, W the image width, and s = 180/W cm per pixel. The implementation estimates width as ws and height as hs/0.707, applying a fixed foreshortening correction. Area is approximated by an ellipse, A = (π/4)ab, and scaled by the square root of detection confidence."),
        P("Depth is estimated from the geometric mean of the approximate dimensions, a shape penalty for elongated boxes, fixed shadow intensity, and a confidence factor. The value is clamped to 1–20 cm. These formulas are deterministic rules, not depth sensing. Without camera calibration, camera pose, known road-plane geometry, or measured pothole depths, the centimeter output has no established physical accuracy."),
        P("The Pothole Severity Index (PSI) is a piecewise score using maximum estimated depth and total estimated area, then scaled by mean confidence and clamped to 0–100. Thresholds assign P1, P2, P3, or normal priority, with a text advisory, suggested material, and cost estimate. Cracks can contribute to the report when no potholes are detected. Code comments refer to ASTM/IRC practices, but the repository does not include a calibration study or demonstrated compliance assessment. PSI should therefore be described as a project-defined score, not a standards-certified index."),
        SeverityFlow(), P("Fig. 3. Rule-based severity flow; physical scale is assumed rather than calibrated.", "Caption"),
        P("IV. IMPLEMENTATION", "Section"),
        P("The API is implemented with FastAPI, which supports uploaded file handling through UploadFile [5], and Starlette sessions. Email/password accounts are stored in SQLite, with passwords processed using salted scrypt hashes. Google OAuth is optional and depends on local credentials. Static assets provide a progressive web application shell. The browser uses IndexedDB for scan history and Leaflet for map rendering. Streamlit provides a second dashboard and uses a cached detector instance. Model loading is lazy at application initialization; the model weights are included in the repository."),
        P("The detector thresholds are mutable instance attributes. In the API endpoint, request parameters update these shared attributes before inference; concurrent requests could therefore observe another request's thresholds. The telemetry route reads a CSV file and returns the requested leading rows. For deployment, input limits, authorization policy, concurrent inference, storage lifecycle, secret management, and API rate controls require further review."),
        P("V. EVALUATION STATUS", "Section"),
        P("The checked-in evaluation_metrics.json and yolo_metrics.json contain precision, recall, mAP, latency, and model-size fields. However, Code/model/train_eval.py writes hard-coded example values and does not load a labeled test set, run model predictions, or compute these statistics. Although RDD2022 is a published multi-national road-damage dataset [2], the repository does not establish that its checked-in metrics were computed on that dataset or provide an auditable evaluation protocol tying those values to annotations and model outputs. These numbers are consequently excluded as verified experimental results."),
        P("A valid evaluation should define a disjoint test split and provide image-level ground-truth boxes for every reported class. Predictions should be scored at stated IoU thresholds, with per-class precision, recall, AP, and mAP computed by a standard implementation. Latency should be measured on named hardware with preprocessing, inference, and postprocessing clearly delineated. Severity should be evaluated against field measurements using absolute error and calibration analyses. Repair-cost estimates should be compared with documented local schedules of rates."),
        P("The current repository includes four auth-route unit tests, but this report does not assert a completed detector benchmark. A reproducible experiment must archive dataset provenance, split identifiers, model hash, configuration, library versions, and generated prediction files."),
        NextPageTemplate("Gallery"), PageBreak(),
        P("VIII. APPLICATION INTERFACE", "Section"),
        P("The browser application presents account creation and sign-in before opening the inspection workspace. After authentication, users can upload an image, review the annotated result, inspect severity fields, and browse device-local scan history. Fig. 4 and Fig. 5 show the actual application screens captured from the running local build. The public deployment is available at <link href='https://pothole-detection-systemm.onrender.com' color='#24583b'>pothole-detection-systemm.onrender.com</link>; its API health endpoint reported the service online with model mode during report preparation."),
        Spacer(1, 5),
        Table([[Image(str(ROOT/'Code'/'report'/'figures'/'roadguard_login.png'), width=3.45*inch, height=2.40*inch), Image(str(ROOT/'Code'/'report'/'figures'/'roadguard_workspace.png'), width=3.45*inch, height=2.40*inch)]], colWidths=[3.58*inch,3.58*inch], style=TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("ALIGN",(0,0),(-1,-1),"CENTER"),("BOX",(0,0),(-1,-1),.4,colors.HexColor("#c5ceca")),("INNERGRID",(0,0),(-1,-1),.4,colors.HexColor("#c5ceca")),("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)])),
        Table([[P("Fig. 4. RoadGuard signup and sign-in screen.","Caption"),P("Fig. 5. Authenticated inspection workspace.","Caption")]], colWidths=[3.58*inch,3.58*inch]),
        P("Interface labels describe the intended workflow. Scan history is local to the browser and is not synchronized to the signed-in account. The workspace screenshot shows the initial empty state; inspection results depend on the selected input image and available model weights.", "SmallIEEE"),
        Spacer(1, 10),
        P("IMPLEMENTED WORKFLOW AT A GLANCE", "Section"),
        Table([
            [P("Stage","SmallIEEE"),P("Project behavior","SmallIEEE")],
            [P("Identity","SmallIEEE"),P("Email/password accounts use a local SQLite database; Google OAuth is optional.","SmallIEEE")],
            [P("Inspection","SmallIEEE"),P("JPEG, PNG, and WebP uploads are checked before model inference or image-processing fallback.","SmallIEEE")],
            [P("Review","SmallIEEE"),P("Annotated image, detections, severity fields, priority, and advisory are shown in the workspace.","SmallIEEE")],
            [P("History","SmallIEEE"),P("Scan records remain in browser storage; location and CSV/print export are optional.","SmallIEEE")],
        ], colWidths=[1.0*inch,6.0*inch], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf4ef")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#24583b")),
            ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#c5ceca")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ])),
        Spacer(1,10), PriorityChart(),
        P("Fig. 6. Counts of priority labels in the bundled CSV. These are descriptive sample records, not detector results independently reproduced for this report.", "Caption"),
        P("The sample contains 28 P1, 14 P2, 7 P3, and 1 normal record. This skew helps demonstrate triage and map views, but it cannot establish defect prevalence or model performance. The CSV records have no linked image-and-annotation set in the repository and are not used as a detector test set.", "SmallIEEE"),
        NextPageTemplate("TwoCol"), PageBreak(),
        P("IX. IMPLEMENTATION DETAILS AND VALIDATION PLAN", "Section"),
        P("The API boundary separates image transport from the browser presentation. The key routes and their current responsibilities are summarized in Table I. The detector response contains model-produced boxes where model inference is active, while severity and maintenance fields are derived by project rules. An image-processing fallback may produce candidate regions when the learned detector is unavailable; the health endpoint exposes the active mode."),
        Table([
            [P("Route","SmallIEEE"),P("Purpose","SmallIEEE")],
            [P("GET /api/v1/health","SmallIEEE"),P("Service status and active detection mode.","SmallIEEE")],
            [P("POST /api/v1/detect","SmallIEEE"),P("Validate image, infer detections, estimate severity, return annotated image.","SmallIEEE")],
            [P("GET /api/v1/telemetry","SmallIEEE"),P("Read the bundled CSV and return up to the requested number of rows.","SmallIEEE")],
            [P("POST /auth/register, /auth/login","SmallIEEE"),P("Create a local account or verify credentials and start a session.","SmallIEEE")],
            [P("GET /auth/google/login","SmallIEEE"),P("Begin optional Google OAuth when credentials are configured.","SmallIEEE")],
        ], colWidths=[1.8*inch, COL_W-1.8*inch], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf4ef")),
            ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#c5ceca")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])),
        Spacer(1,4), P("TABLE I. MAIN API ROUTES IN THE CURRENT PROTOTYPE", "Caption"),
        P("A. Reproducible detector evaluation", "Subsection"),
        P("A future detector benchmark should freeze the model artifact and test data before running inference. Each image must have independently checked annotations for Pothole, Road_Crack, and Manhole_Drain, with a documented policy for ambiguous or partially visible defects. Split assignment must be performed at the road or capture-sequence level where possible so near-duplicate frames do not leak across training and test sets."),
        FrameBreak(),
        P("For each class, report precision, recall, and average precision at a stated IoU threshold; report mAP across the selected IoU range and explain the averaging convention. Publish confidence and NMS settings, image resize policy, package versions, hardware, batch size, and warm-up procedure. Timing should be repeated over enough images to report median and percentile latency, and should identify whether file decoding and annotation rendering are included."),
        P("B. Severity and cost validation", "Subsection"),
        P("Depth estimates require a reference measurement protocol, such as a calibrated depth sensor or surveyed profile. Compare predicted and measured depth on held-out sites and report mean absolute error, root mean square error, and error by camera distance and lighting condition. Area estimates should be compared with manually delineated ground-plane regions. Priority classification should be evaluated against a defined expert-reviewed rubric, with disagreement cases reported."),
        P("Repair-cost estimates need dated, regional schedule-of-rate sources and a transparent calculation for quantities, labor, mobilization, and taxes. Until such evidence is added, the displayed cost should be labeled illustrative. The bundled telemetry should also be documented with provenance, collection method, consent, and data-quality checks before it is used as empirical evidence."),
        P("C. Operational checks", "Subsection"),
        P("Before deployment, test malformed and oversized uploads, unsupported formats, empty files, model-load failure, concurrent requests with distinct thresholds, expired sessions, duplicate account registration, and logout behavior. Apply an explicit bound to telemetry limits. Deployment checks should include HTTPS-only cookies, a stable secret key, exact OAuth redirect URIs, database backup and retention, and a privacy review for GPS-bearing scan records."),
        P("These checks form a proposed validation plan; they are not all completed experiments. The current test suite covers core auth routes, while detector accuracy, depth calibration, concurrency behavior, and operational load remain open validation items."),
        NextPageTemplate("TwoCol"), PageBreak(),
        P("VI. LIMITATIONS AND ETHICS", "Section"),
        P("The system processes single still images; performance across weather, motion blur, camera height, road materials, and countries has not been demonstrated. Dark-region fallback logic can confuse shadows, stains, and road openings with defects. Bounding boxes can include substantial background, while fixed pixel-to-centimeter scaling ignores perspective variation. Confidence is not a physical measurement. Priority rules and repair prices are project heuristics, not engineering recommendations validated by a road authority."),
        P("Location capture and stored inspection images can reveal sensitive information. The browser stores history on-device, and server-side account data is local to a SQLite file. A production deployment needs clear retention controls, access control, HTTPS, protected secrets, and privacy review. Google sign-in only establishes identity; scan history is not account-synchronized in the current browser implementation."),
        P("VII. CONCLUSION", "Section"),
        P("RoadGuard integrates image upload, object detection, heuristic severity scoring, API delivery, and inspection-oriented interfaces into a cohesive prototype. Its main value is the complete workflow and the way results are made reviewable. The present evidence does not establish detector accuracy, calibrated depth, standards compliance, or reliable repair cost. The next steps are to create a reproducible labeled test set, calibrate the imaging geometry, validate severity against physical measurements, and replace illustrative telemetry and cost assumptions with traceable data."),
        P("IMPLEMENTATION ARTIFACTS", "Subsection"),
        Table([
            [P("Source module","SmallIEEE"),P("Implemented role","SmallIEEE")],
            [P("app/main.py","SmallIEEE"),P("API routes, sessions, account flow, upload checks.","SmallIEEE")],
            [P("model/pothole_detector.py","SmallIEEE"),P("YOLO model loading and image-processing fallback.","SmallIEEE")],
            [P("model/severity_engine.py","SmallIEEE"),P("Pixel-based estimates, PSI rules, priority text.","SmallIEEE")],
            [P("app/static/*","SmallIEEE"),P("Browser UI, local history, map and exports.","SmallIEEE")],
            [P("app/streamlit_app.py","SmallIEEE"),P("Separate inspection and telemetry dashboard.","SmallIEEE")],
        ], colWidths=[1.65*inch,COL_W-1.65*inch], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf4ef")),
            ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#c5ceca")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])),
        Spacer(1,4), P("TABLE II. PRINCIPAL SOURCE MODULES", "Caption"),
        P("EVIDENCE AVAILABLE IN THE REPOSITORY", "Subsection"),
        Table([
            [P("Claim area","SmallIEEE"),P("Current evidence","SmallIEEE")],
            [P("Detection accuracy","SmallIEEE"),P("Metric JSON only; evaluation script does not compute predictions.","SmallIEEE")],
            [P("Depth and area","SmallIEEE"),P("Fixed-scale heuristic; no calibration or surveyed ground truth.","SmallIEEE")],
            [P("Telemetry","SmallIEEE"),P("50 CSV rows; source and image links are not established.","SmallIEEE")],
            [P("Authentication","SmallIEEE"),P("Four route tests cover the local auth flow.","SmallIEEE")],
        ], colWidths=[1.22*inch,COL_W-1.22*inch], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf4ef")),
            ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#c5ceca")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])),
        FrameBreak(), P("REFERENCES", "Section"),
        P("[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” in <i>Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)</i>, 2016, pp. 779–788, doi: 10.1109/CVPR.2016.91.", "Ref"),
        P("[2] A. Arya et al., “RDD2022: A multi-national image dataset for automatic Road Damage Detection,” arXiv:2209.08538, 2022.", "Ref"),
        P("[3] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” <i>IEEE Trans. Pattern Anal. Mach. Intell.</i>, vol. 39, no. 6, pp. 1137–1149, 2017, doi: 10.1109/TPAMI.2016.2577031.", "Ref"),
        P("[4] Ultralytics, “Predict Mode,” Ultralytics Documentation. [Online]. Available: https://docs.ultralytics.com/modes/predict. Accessed: Nov. 21, 2026.", "Ref"),
        P("[5] FastAPI, “Request Files,” FastAPI Documentation. [Online]. Available: https://fastapi.tiangolo.com/tutorial/request-files/. Accessed: Nov. 21, 2026.", "Ref"),
        P("[6] RoadGuard source repository, “Pothole-detection-systemm,” GitHub. [Online]. Available: https://github.com/Devanshu-Dhyanu/Pothole-detection-systemm. Accessed: Nov. 21, 2026.", "Ref"),
        P("[7] RoadGuard, “Browser-based Road Inspection Application,” Render. [Online]. Available: https://pothole-detection-systemm.onrender.com. Accessed: Nov. 21, 2026.", "Ref"),
        P("VALIDATION STATUS", "Subsection"),
        P("Implemented in the repository: upload validation, model/fallback inference paths, rule-based severity output, browser-local scan history, telemetry display, and core auth-route tests. Not established by this report: independently computed detection metrics, calibrated depth accuracy, ground-truth severity agreement, repair-cost accuracy, or field performance. This distinction keeps sample records and configured user flows separate from completed validation.", "SmallIEEE"),
        P("RECOMMENDED NEXT MILESTONES", "Subsection"),
        Table([
            [P("Milestone","SmallIEEE"),P("Evidence to produce","SmallIEEE")],
            [P("Dataset","SmallIEEE"),P("Provenance, split manifest, reviewed boxes, class definitions.","SmallIEEE")],
            [P("Model benchmark","SmallIEEE"),P("Computed AP/precision/recall, configs, model hash, hardware timings.","SmallIEEE")],
            [P("Physical severity","SmallIEEE"),P("Calibrated camera geometry and held-out depth/area measurements.","SmallIEEE")],
            [P("Field readiness","SmallIEEE"),P("Expert triage review, current repair rates, privacy and security checks.","SmallIEEE")],
        ], colWidths=[1.15*inch,COL_W-1.15*inch], style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf4ef")),
            ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#c5ceca")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ])),
    ]
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
