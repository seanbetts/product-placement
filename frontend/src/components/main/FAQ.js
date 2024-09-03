import React from 'react';
import { Typography, Accordion, AccordionSummary, AccordionDetails, Box } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const faqData = [
  {
    question: "What is Product Placement Detection?",
    answer: "Product Placement Detection is a tool that analyzes videos to identify and track products or brands that appear within the content. It uses advanced AI and machine learning techniques to detect, recognize, and report on product placements in videos."
  },
  {
    question: "How does the video processing work?",
    answer: "Our system processes uploaded videos in several steps: video frame extraction, audio extraction, transcription generation, and optical character recognition (OCR). These processes allow us to analyze the visual, audio, and textual content of the video to identify product placements."
  },
  {
    question: "What file formats are supported for video upload?",
    answer: "We support most common video formats, including MP4, AVI, MOV, and WMV. For the best results, we recommend using MP4 files with H.264 video codec and AAC audio codec."
  },
  {
    question: "How long does it take to process a video?",
    answer: "Processing time depends on the length and complexity of the video. Typically, it takes about 1-2 times the duration of the video. For example, a 10-minute video might take 10-20 minutes to process fully."
  },
  {
    question: "Can I download the results of the analysis?",
    answer: "Yes, you can download various outputs from the analysis, including the processed video, audio, transcript, and a word cloud representing the detected text and brands."
  }
];

const FAQ = () => {
  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Frequently Asked Questions - WIP
      </Typography>
      {faqData.map((faq, index) => (
        <Accordion key={index}>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls={`panel${index + 1}-content`}
            id={`panel${index + 1}-header`}
          >
            <Typography fontWeight="bold">{faq.question}</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography>{faq.answer}</Typography>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
};

export default FAQ;