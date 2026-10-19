/** Decode actual raster bytes, preserving aspect ratio/alpha; never accept URLs or SVG. */
export async function decodeBrandingImage(file: File): Promise<string> {
  if (file.size > 2 * 1024 * 1024 || file.size === 0) {
    throw new Error('size')
  }

  const bytes = new Uint8Array(await file.arrayBuffer())
  const ascii = (start: number, end: number) => String.fromCharCode(...bytes.slice(start, end))
  const png = [137, 80, 78, 71, 13, 10, 26, 10].every((value, i) => bytes[i] === value)
  const jpeg = bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255
  const webp = ascii(0, 4) === 'RIFF' && ascii(8, 12) === 'WEBP'

  if (!png && !jpeg && !webp) {
    throw new Error('format')
  }

  const image = await createImageBitmap(file)

  try {
    if (image.width < 32 || image.height < 32 || image.width > 4096 || image.height > 4096) {
      throw new Error('dimensions')
    }

    const canvas = document.createElement('canvas')
    canvas.width = image.width
    canvas.height = image.height
    const context = canvas.getContext('2d')

    if (!context) {
      throw new Error('decode')
    }

    context.drawImage(image, 0, 0)
    const result = canvas.toDataURL('image/png')

    if (result.length > Math.ceil((2 * 1024 * 1024) / 3) * 4 + 32) {
      throw new Error('size')
    }

    return result
  } finally {
    image.close()
  }
}
