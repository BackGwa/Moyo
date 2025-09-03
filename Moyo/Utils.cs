using Microsoft.ML.OnnxRuntime;

namespace Moyo
{
    public class Metadata
    {
        Dictionary<string, string> metadata;

        public Metadata(ModelMetadata modelMetadata)
        {
            metadata = modelMetadata.CustomMetadataMap;
        }

        public ModelType ParseType()
        {
            if (!metadata.ContainsKey("task")) return ModelType.UnknownModel;

            switch (metadata["task"])
            {
                case "detect": return ModelType.ObjectDetection;
                case "segment": return ModelType.Segmentation;
                case "classify": return ModelType.Classification;
                case "pose": return ModelType.PoseEstimation;
                case "obb": return ModelType.OrientedBoundingBox;
                default: return ModelType.UnknownModel;
            }
        }

        public string[] ParseLabels()
        {
            if (!metadata.ContainsKey("names")) return new string[] { };

            var names = new List<string>();
            string cleaned = metadata["names"].Trim('{', '}', ' ');
            var pairs = cleaned.Split(',');

            foreach (var pair in pairs)
            {
                var keyValue = pair.Split(':');
                if (keyValue.Length == 2)
                {
                    string value = keyValue[1].Trim().Trim('\'', '"', ' ');
                    names.Add(value);
                }
            }
            return names.ToArray();
        }
    }

    public static class Utils
    {
        public static float BoundingLimit(float value, float max)
        {
            return (value < 0) ? 0 : (value > max) ? max : value;
        }

        public static float ComputeIoU(ObjectRect boxA, ObjectRect boxB) {
            float interXmin = Math.Max(boxA.xMin, boxB.xMin);
            float interYmin = Math.Max(boxA.yMin, boxB.yMin);
            float interXmax = Math.Min(boxA.xMax, boxB.xMax);
            float interYmax = Math.Min(boxA.yMax, boxB.yMax);

            float interWidth = Math.Max(0, interXmax - interXmin);
            float interHeight = Math.Max(0, interYmax - interYmin);
            float interArea = interWidth * interHeight;

            float boxAArea = (boxA.xMax - boxA.xMin) * (boxA.yMax - boxA.yMin);
            float boxBArea = (boxB.xMax - boxB.xMin) * (boxB.yMax - boxB.yMin);

            return interArea == 0 ? 0 : interArea / (boxAArea + boxBArea - interArea);
        }
    }
}