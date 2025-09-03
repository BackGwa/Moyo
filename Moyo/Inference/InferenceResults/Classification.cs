namespace Moyo
{
    public partial class InferenceResult
    {
        public Top1? top1 { get; set; }
        public Top5? top5 { get; set; }
    }

    public class Top1
    {
        public required int classes { get; set; }
        public required float confidence { get; set; }
    }

    public class Top5
    {
        public required int[] classes { get; set; }
        public required float[] confidences { get; set; }
    }
}