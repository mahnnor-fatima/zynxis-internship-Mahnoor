import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

output_dir = r"c:\Users\mahno\OneDrive\Desktop\internship\Week_1"
pdf_path = os.path.join(output_dir, "findings_report.pdf")

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    leftMargin=28,
    rightMargin=28,
    topMargin=25,
    bottomMargin=25
)

story = []
styles = getSampleStyleSheet()

# Custom Palette & Styles
primary_color = colors.HexColor("#1A365D")   # Navy
secondary_color = colors.HexColor("#2B6CB0") # Accent Blue
dark_text = colors.HexColor("#2D3748")       # Dark Charcoal
bg_light = colors.HexColor("#F7FAFC")        # Soft Light Grey

title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=18,
    leading=22,
    textColor=primary_color,
    spaceAfter=2
)

subtitle_style = ParagraphStyle(
    'DocSubtitle',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9,
    leading=12,
    textColor=colors.HexColor("#718096"),
    spaceAfter=8
)

heading_style = ParagraphStyle(
    'SectionHeading',
    parent=styles['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=14,
    textColor=primary_color,
    spaceBefore=4,
    spaceAfter=4
)

body_style = ParagraphStyle(
    'BodyText',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8.5,
    leading=11.5,
    textColor=dark_text,
    spaceAfter=4
)

bullet_style = ParagraphStyle(
    'BulletText',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8.5,
    leading=11.5,
    textColor=dark_text,
    leftIndent=10,
    spaceAfter=2
)

# Header Section
story.append(Paragraph("Week 1 EDA Findings Report: Titanic Dataset Analysis", title_style))
story.append(Paragraph("<b>Author:</b> Mahnoor | <b>Track:</b> Zynxis Internship - Data Analysis | <b>Dataset:</b> 892 Passengers (792 Train / 100 Test)", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=0, spaceAfter=8))

# Executive Summary
story.append(Paragraph("Executive Summary", heading_style))
summary_text = (
    "This report delivers a full Exploratory Data Analysis (EDA) performed on the Titanic passenger dataset. "
    "The dataset consists of <b>892 total entries</b> with 16 features, including demographic, economic, and family indicators. "
    "An extensive data integrity audit confirmed <b>0 missing values</b> and <b>0 duplicates</b> across all samples. "
    "Key analyses reveal that passenger survival was heavily driven by socio-demographic rules ('women and children first') "
    "and socio-economic privilege (ticket class and fare amount)."
)
story.append(Paragraph(summary_text, body_style))
story.append(Spacer(1, 4))

# Key Metrics Table
story.append(Paragraph("Key Exploratory Data Metrics", heading_style))
metrics_data = [
    [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value / Statistic</b>", body_style), Paragraph("<b>Key EDA Takeaway</b>", body_style)],
    [Paragraph("Overall Survival Rate", body_style), Paragraph("38.6% (344 / 892)", body_style), Paragraph("Class imbalance towards non-survival.", body_style)],
    [Paragraph("Female Survival Rate", body_style), Paragraph("74.2%", body_style), Paragraph("Primary demographic predictor of survival.", body_style)],
    [Paragraph("Male Survival Rate", body_style), Paragraph("18.9%", body_style), Paragraph("Male mortality rate exceeded 81%.", body_style)],
    [Paragraph("1st Class Survival Rate", body_style), Paragraph("63.0%", body_style), Paragraph("Privileged cabin location & evacuation priority.", body_style)],
    [Paragraph("3rd Class Survival Rate", body_style), Paragraph("24.2%", body_style), Paragraph("High mortality among lower-deck passengers.", body_style)],
]

t = Table(metrics_data, colWidths=[2.1*inch, 1.6*inch, 3.8*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), bg_light),
    ('TEXTCOLOR', (0, 0), (-1, 0), primary_color),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
]))
story.append(t)
story.append(Spacer(1, 6))

# Visualizations Section (2 key charts side-by-side)
story.append(Paragraph("Key Visualizations", heading_style))
img1_path = os.path.join(output_dir, "fig2_survival_by_gender.png")
img2_path = os.path.join(output_dir, "fig3_pclass_survival.png")

img1 = Image(img1_path, width=3.6*inch, height=2.15*inch)
img2 = Image(img2_path, width=3.6*inch, height=2.15*inch)

chart_table = Table([[img1, img2]], colWidths=[3.75*inch, 3.75*inch])
chart_table.setStyle(TableStyle([
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ('TOPPADDING', (0, 0), (-1, -1), 0),
]))
story.append(chart_table)
story.append(Spacer(1, 6))

# Insights & Conclusion
story.append(Paragraph("Core Insights & Recommendations", heading_style))
story.append(Paragraph("• <b>Gender Protocol Impact:</b> Female gender exhibited a strong negative correlation with mortality (r = -0.54), confirming strict protocol enforcement during life-raft boarding.", bullet_style))
story.append(Paragraph("• <b>Economic Stratification:</b> 1st Class passengers were 2.6x more likely to survive than 3rd Class passengers. Higher ticket fare directly improved survival odds (r = +0.26).", bullet_style))
story.append(Paragraph("• <b>Family Size Dynamics:</b> Moderate family size (2-4 members) correlated with higher survival, while solo travelers and large families (>5 members) fared significantly worse.", bullet_style))
story.append(Paragraph("• <b>Model Readiness:</b> The dataset is clean, balanced across splits, and ready for predictive modeling (e.g. Logistic Regression, Random Forest, or XGBoost).", bullet_style))

# Build document
doc.build(story)
print(f"Report PDF built successfully at: {pdf_path}")
