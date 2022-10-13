from tape import TAPETokenizer, ProteinBertModel
import re
import torch
import random
            
AALETTER = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

# add 'X/U/O*' to the list as unknown or missing amminoacid
# reference: https://wiki.thegpm.org/wiki/Amino_acid_symbols
AALETTER_REGEX = "^[" + "".join(AALETTER) + "XUO*" + "]*$"

def strip_sequence(seq):
    seqs = seq.strip().split("\n")
    ret = ""
    for s in seqs:
        if not s.startswith(">"):
            ret += s
    return ret

def extract_sequence(seq):
    if "\n" in seq:
        seq = strip_sequence(seq)
        if not re.match(AALETTER_REGEX, seq):
            raise ValueError("Protein sequence contains invalid letter(s):" + seq)
    return seq


bert_model = None
tokenizer = TAPETokenizer(vocab="iupac")
BERT_FEATURE_LENGTH = 768

def encode_sequence_bert(sequence, device="cpu"):
    global bert_model
    if not bert_model:
        bert_model = ProteinBertModel.from_pretrained("bert-base")
        bert_model.eval()
        bert_model.to(device)
    # tokenizer = TAPETokenizer(vocab="unirep")
    protein_sequence = extract_sequence(sequence) # length: 932
    vec = tokenizer.encode(protein_sequence) # length: 934
    protein_vector = torch.tensor(vec[None, :]).to(device)
    with torch.no_grad():
        local_embedding, global_embedding = bert_model(protein_vector) # torch.Size([1, 933, 768]), torch.Size([1, 768])
    return local_embedding[0], global_embedding[0]


if __name__ == "__main__":
    sequence = '>sp|Q9Y5G6|PCDG7_HUMAN Protocadherin gamma-A7 OS=Homo sapiens OX=9606 GN=PCDHGA7 PE=2 SV=1\nMAAQPRGGDYRGFFLLSILLGTPWEAWAGRILYSVSEETDKGSFVGDIAKDLGLEPRELA\nERGVRIISRGRTQLFALNQRSGSLVTAGRIDREEICAQSARCLVNFNILMEDKMNLYPID\nVEIIDINDNVPRFLTEEINVKIMENTAPGVRFPLSEAGDPDVGTNSLQSYQLSPNRHFSL\nAVQSGDDETKYPELVLERVLDREEERVHHLVLTASDGGDPPRSSTAHIQVTVVDVNDHTP\nVFSLPQYQVTVPENVPVGTRLLTVHAIDLDEGVNGEVTYSFRKITPKLPKMFHLNSLTGE\nISTLEGLDYEETAFYEMEVQAQDGPGSLTKAKVLITVLDVNDNAPEVTMTSLSSSIPEDT\nPLGTVIALFYLQDRDSGKNGEVTCTIPENLPFKLEKSIDNYYRLVTTKNLDRETLSLYNI\nTLKATDGGTPPLSRETHIFMQVADTNDNPPTFPHSSYSVYIAENNPRGASIFLVTAQDHD\nSEDNAQITYSLAEDTIQGAPVSSYVSINSDTGVLYALQSFDYEQLRELQLRVTAHDSGDP\nPLSSNMSLSLFVLDQNDNPPEILYPALPTDGSTGMELAPRSAEPGYLVTKVVAVDKDSGQ\nNAWLSYLLLKASEPGLFAVGLYTGEVRTARALLDRDALKQSLVVAVQDHGQPPLSATVTL\nTVAVADSIPEVLADLGSLEPSDGPYNYDLTLYLVVAVATVSCVFLAFVLVLLALRLRRWH\nKSRLLQASEGGLANVPTSHFVGMDGVQAFLQTYSHEVSLTADSRKSHLIFPQPNYVDMLI\nSQESCEKNDSLLTSVDFQECKENLPSIQQAPPNTDWRFSQAQRPGTSGSQNGDDTGTWPN\nNQFDTEMLQAMILASASEAADGSSTLGGGAGTMGLSARYGPQFTLQHVPDYRQNVYIPGS\nNATLTNAAGKRDGKAPAGGNGNKKKSGKKEKK\n'
    _, embed = encode_sequence_bert(sequence)
    print(embed.shape)