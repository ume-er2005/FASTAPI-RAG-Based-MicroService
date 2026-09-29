from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_sample_course_pdf(filename="CS402_Artificial_Intelligence_Syllabus.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1D4ED8'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=6
    )

    story = []

    # --- PAGE 1: Course Info & Grading ---
    story.append(Paragraph("CS402: Advanced Artificial Intelligence & RAG Systems", title_style))
    story.append(Paragraph("<b>Department of Computer Science | Fall Semester</b>", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Course Overview", heading_style))
    story.append(Paragraph(
        "This course explores foundational and modern principles of Artificial Intelligence, "
        "natural language processing, and neural retrieval systems. Students will delve into search algorithms, "
        "embeddings, dense vector retrieval, and Retrieval-Augmented Generation (RAG). "
        "Practical laboratory sessions will focus on FastAPI backend integration and LLM application development.",
        body_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Instructor & Teaching Assistants", heading_style))
    story.append(Paragraph(
        "<b>Lead Instructor:</b> Dr. Sarah Jenkins (Email: s.jenkins@university.edu)<br/>"
        "<b>Office Hours:</b> Tuesday & Thursday, 2:00 PM – 4:00 PM (Room 412, Turing Hall)<br/>"
        "<b>Teaching Assistant:</b> Alex Rivera (Email: a.rivera@university.edu)<br/>"
        "<b>TA Lab Hours:</b> Monday & Wednesday, 10:00 AM – 12:00 PM (CS Lab 3)",
        body_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Grading Breakdown & Criteria", heading_style))
    story.append(Paragraph(
        "The course grading policy is structured to emphasize both theoretical understanding and hands-on implementation.",
        body_style
    ))
    
    grading_data = [
        ["Assessment Category", "Weight", "Description"],
        ["Programming Assignments (4x)", "25%", "FastAPI services, vector indexing, and chunking"],
        ["Midterm Exam", "20%", "Foundational theory, search algorithms, probability"],
        ["Course RAG Capstone Project", "30%", "End-to-end PDF Q&A RAG application with evaluation"],
        ["Class Participation & Quizzes", "10%", "Weekly comprehension quizzes and active discussion"],
        ["Final Examination", "15%", "Comprehensive theoretical and applied concepts"]
    ]
    t = Table(grading_data, colWidths=[160, 70, 270])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E0E7FF')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Late Submission & Attendance Policy", heading_style))
    story.append(Paragraph(
        "Assignments submitted after the deadline incur a 10% penalty per day up to a maximum of 3 days. "
        "Submissions beyond 72 hours past the deadline receive a grade of zero unless an official medical extension "
        "is approved prior to the due date. Each student is granted 2 slip days to be used throughout the semester for emergencies.",
        body_style
    ))

    # --- PAGE 2: Weekly Schedule & Capstone Details ---
    story.append(PageBreak())

    story.append(Paragraph("Weekly Schedule & Course Modules", heading_style))
    
    schedule_data = [
        ["Week", "Topic", "Core Readings & Deliverables"],
        ["Week 1-2", "Introduction to AI & Python Environment Setup", "Search algorithms, heuristics (A*). Assignment 1 released."],
        ["Week 3-4", "Vector Spaces & Text Embeddings", "Cosine similarity, TF-IDF vs Dense vectors, semantic search."],
        ["Week 5-6", "Vector Databases & Indexing Strategies", "Chunking strategies, metadata tagging, ChromaDB/FAISS. Assignment 2."],
        ["Week 7", "Midterm Review & Examination", "Covers Weeks 1-6."],
        ["Week 8-9", "Large Language Models & Prompt Engineering", "Gemini API, zero-shot/few-shot, hallucinations and grounding."],
        ["Week 10-11", "RAG Pipeline Architecture & Evaluation", "Retrieval augmented generation, reranking, RAG triad. Assignment 3."],
        ["Week 12-13", "FastAPI Production Deployment", "Asynchronous endpoints, Dockerization, Swagger API docs."],
        ["Week 14", "Capstone Project Presentations", "Live demo of PDF RAG system and oral defense."]
    ]
    t2 = Table(schedule_data, colWidths=[60, 200, 240])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E0E7FF')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Capstone Project Requirements", heading_style))
    story.append(Paragraph(
        "For the 30% capstone project, students must build a complete RAG system that accepts PDF uploads, "
        "chunks and indexes the documents, and allows interactive Q&A powered by an LLM. "
        "The project must be served via FastAPI with clean endpoint documentation at /docs. "
        "Citing exact page numbers and document sources is mandatory to prevent hallucinations.",
        body_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Academic Integrity Policy", heading_style))
    story.append(Paragraph(
        "All code and deliverables must be the student's individual work. While using generative AI tools as a learning "
        "assistant is encouraged, copying another student's submission or directly copying unverified code constitutes academic "
        "misconduct and will be referred to the University Academic Board.",
        body_style
    ))

    doc.build(story)
    print(f"Generated sample PDF at: {filename}")

if __name__ == "__main__":
    create_sample_course_pdf()
