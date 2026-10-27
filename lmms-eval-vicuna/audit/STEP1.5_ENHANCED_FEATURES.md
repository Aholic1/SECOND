# Step 1.5: Enhanced Visual Feature Extraction

## Summary

Step 1.5 enhances the baseline data with detailed visual features extracted from COCO annotations, enabling fine-grained stratification by:
- Target object size (small/medium/large)
- Target object position (center/peripheral)
- Image complexity (simple/medium/complex)

## Key Improvements Over Step 1

### Before (Step 1)
- Coarse stratification based on question keywords only
- 6 visual strata: TP_simple, TP_relation, TN_correct, FN_simple, FN_relation, FP_hallucination
- 100 diagnostic samples (78 after filtering)
- No ground-truth object localization

### After (Step 1.5)
- Fine-grained stratification using COCO bounding boxes
- 13 visual strata covering size, position, and complexity dimensions
- 120 diagnostic samples with balanced coverage
- 85.8% target localization rate (1355/1580 non-TN samples)

## Output Files

### Enhanced Features
**Location:** `audit/step1_full/pope_pop_3000_baseline/enhanced_features.jsonl`

Each record includes:
```json
{
  "question_id": "887",
  "stratum": "FN",
  "visual_stratum": "FN_small_peripheral",
  "image_id": "3886",
  "gt": "yes",
  "pred": "no",
  "question": "Is there a bird in the image?",
  
  "num_objects": 15,
  "complexity": "complex",
  "image_width": 640,
  "image_height": 480,
  
  "target_category": "bird",
  "target_bbox": [320, 50, 40, 45],
  "target_area_ratio": 0.0058,
  "target_size": "small",
  "target_position": "peripheral",
  "distance_to_center_normalized": 0.42
}
