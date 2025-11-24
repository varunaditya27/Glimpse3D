# Pipeline Flow

Detailed explanation of the data flow through the Glimpse3D system.

1.  **Input**: User uploads an image.
2.  **Preprocessing**: Background removal, resizing.
3.  **Coarse Gen**: TripoSR generates initial mesh/splats.
4.  **Refinement Loop**:
    *   Render view
    *   Enhance (SDXL)
    *   Back-project (Update Splats)
5.  **Export**: Final model conversion.
