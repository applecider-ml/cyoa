# CYOA SkyPortal Integration Notes for PR

## 1. Widget Component (\CYOAWidget.jsx\)
Place this file into: \static/js/components/source/CYOAWidget.jsx\

## 2. Parent Layout Modification (\Source.jsx\)
Open \static/js/components/source/Source.jsx\ in the main repository.

Add this import at the top:
\\\javascript
import CYOAWidget from './CYOAWidget';
\\\

Locate the \Centroid Plot\ Accordion block and insert the CYOAWidget directly above it as follows:
\\\javascript
      <Grid item xs={12} lg={12} order={{ xs: 14, md: 11, lg: 9 }}>
        <Accordion
          defaultExpanded
          disableGutters
          className={classes.flexColumn}
        >
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls="cyoa-widget-content"
            id="cyoa-widget-header"
          >
            <Typography className={classes.accordionHeading}>
              CYOA Analysis
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <CYOAWidget sourceId={source.id} />
          </AccordionDetails>
        </Accordion>
      </Grid>
\\\

## 3. Ingestion Python Bot
The \push_to_skyportal.py\ file in the root of \5_final_underconstruction\ handles the actual backend push utilizing the newly registered token on SkyPortal/Fritz.
