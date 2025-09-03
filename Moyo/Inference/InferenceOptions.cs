namespace Moyo
{
    public class InferenceOptions
    {
        public float confidence { get; set; } = 0.25f;
        public float iou { get; set; } = 0.7f;
        public int detectLimit { get; set; } = 300;

        public static readonly InferenceOptions DEFAULT = new InferenceOptions
        {
            confidence = 0.25f,
            iou = 0.7f,
            detectLimit = 300
        };
    }
}