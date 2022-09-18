import { Center, Flex, Text } from '@chakra-ui/react';
import { RootState } from '../../app/store';
import { useAppSelector } from '../../app/store';
import HoverableImage from './HoverableImage';

/**
 * Simple image gallery.
 */
const ImageGallery = () => {
  const { images, currentImageUuid } = useAppSelector(
    (state: RootState) => state.gallery
  );

  /**
   * I don't like that this needs to rerender whenever the current image is changed.
   * What if we have a large number of images? I suppose pagination (planned) will
   * mitigate this issue.
   *
   * TODO: Refactor if performance complaints, or after migrating to new API which supports pagination.
   */

  return images.length ? (
    <Flex gap={2} wrap="wrap" pb={2}>
      {[...images].reverse().map((image) => {
        const { uuid } = image;
        const isSelected = currentImageUuid === uuid;
        return (
          <HoverableImage key={uuid} image={image} isSelected={isSelected} />
        );
      })}
    </Flex>
  ) : (
    <Center height={'100%'} position={'relative'}>
      <Text size={'xl'}>No images in gallery</Text>
    </Center>
  );
};

export default ImageGallery;
