from cryptography.fernet import Fernet

fernet = Fernet("I47jl1hqUPug4KbVYd60_zeXhn_IH_ECT3QRGiBxdxo=")

def encrypt(target):
    return fernet.encrypt(target)

def decrypt(target):
    return fernet.decrypt(target)