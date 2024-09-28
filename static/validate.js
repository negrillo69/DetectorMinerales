function validateFile() {
    const fileInput = document.getElementById('image');
    const filePath = fileInput.value;
    const allowedExtensions = /(\.jpg|\.jpeg|\.png|\.gif)$/i;

    if (!allowedExtensions.exec(filePath)) {
        alert('Por favor, sube un archivo .jpg, .jpeg, .png o .gif');
        fileInput.value = '';
        return false;
    }
    return true;
}
