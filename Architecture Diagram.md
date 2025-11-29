Architecture Diagram
Description: The diagram shows the flow from candidate data input → AI processing → evaluation → PDF report generation.
[Candidate Inputs] --> [Streamlit UI] --> [Resume PDF Parsing (PyPDF)]
                                   |--> [Candidate Info Form]
                                   |
                                   v
                             [AI Processing: Gemini LLM]
                                   |--> Question Generation
                                   |--> Answer Scoring
                                   |--> Resume Summarization
                                   |
                                   v
                           [Session State Management]
                                   |
                                   v
                             [Results & Feedback]
                                   |
                                   v
                        [PDF Report Generation (FPDF)]
                                   |
                                   v
                         [Downloadable Report Output]
