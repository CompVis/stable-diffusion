import {
  Center,
  Flex,
  Heading,
  IconButton,
  Link,
  Text,
  Tooltip,
} from '@chakra-ui/react';
import { ExternalLinkIcon } from '@chakra-ui/icons';
import { memo } from 'react';
import { IoArrowUndoCircleOutline } from 'react-icons/io5';
import { useAppDispatch } from '../../../app/store';
import * as InvokeAI from '../../../app/invokeai';
import {
  setCfgScale,
  setGfpganStrength,
  setHeight,
  setImg2imgStrength,
  setInitialImagePath,
  setMaskPath,
  setPrompt,
  setSampler,
  setSeed,
  setSeedWeights,
  setShouldFitToWidthHeight,
  setSteps,
  setUpscalingLevel,
  setUpscalingStrength,
  setWidth,
} from '../../options/optionsSlice';
import promptToString from '../../../common/util/promptToString';
import { seedWeightsToString } from '../../../common/util/seedWeightPairs';
import { FaCopy } from 'react-icons/fa';

type MetadataItemProps = {
  isLink?: boolean;
  label: string;
  onClick?: () => void;
  value: number | string | boolean;
  labelPosition?: string;
};

/**
 * Component to display an individual metadata item or parameter.
 */
const MetadataItem = ({
  label,
  value,
  onClick,
  isLink,
  labelPosition,
}: MetadataItemProps) => {
  return (
    <Flex gap={2}>
      {onClick && (
        <Tooltip label={`Recall ${label}`}>
          <IconButton
            aria-label="Use this parameter"
            icon={<IoArrowUndoCircleOutline />}
            size={'xs'}
            variant={'ghost'}
            fontSize={20}
            onClick={onClick}
          />
        </Tooltip>
      )}
      <Flex direction={labelPosition ? 'column' : 'row'}>
        <Text fontWeight={'semibold'} whiteSpace={'pre-wrap'} pr={2}>
          {label}:
        </Text>
        {isLink ? (
          <Link href={value.toString()} isExternal wordBreak={'break-all'}>
            {value.toString()} <ExternalLinkIcon mx="2px" />
          </Link>
        ) : (
          <Text overflowY={'scroll'} wordBreak={'break-all'}>
            {value.toString()}
          </Text>
        )}
      </Flex>
    </Flex>
  );
};

type ImageMetadataViewerProps = {
  image: InvokeAI.Image;
  styleClass?: string;
};

// TODO: I don't know if this is needed.
const memoEqualityCheck = (
  prev: ImageMetadataViewerProps,
  next: ImageMetadataViewerProps
) => prev.image.uuid === next.image.uuid;

// TODO: Show more interesting information in this component.

/**
 * Image metadata viewer overlays currently selected image and provides
 * access to any of its metadata for use in processing.
 */
const ImageMetadataViewer = memo(
  ({ image, styleClass }: ImageMetadataViewerProps) => {
    const dispatch = useAppDispatch();
    // const jsonBgColor = useColorModeValue('blackAlpha.100', 'whiteAlpha.100');

    const metadata = image?.metadata?.image || {};
    const {
      type,
      postprocessing,
      sampler,
      prompt,
      seed,
      variations,
      steps,
      cfg_scale,
      seamless,
      width,
      height,
      strength,
      fit,
      init_image_path,
      mask_image_path,
      orig_path,
      scale,
    } = metadata;

    const metadataJSON = JSON.stringify(metadata, null, 2);

    return (
      <div className={`image-metadata-viewer ${styleClass}`}>
        <Flex gap={1} direction={'column'} width={'100%'}>
          <Flex gap={2}>
            <Text fontWeight={'semibold'}>File:</Text>
            <Link href={image.url} isExternal>
              {image.url}
              <ExternalLinkIcon mx="2px" />
            </Link>
          </Flex>
          {Object.keys(metadata).length > 0 ? (
            <>
              {type && <MetadataItem label="Generation type" value={type} />}
              {['esrgan', 'gfpgan'].includes(type) && (
                <MetadataItem label="Original image" value={orig_path} />
              )}
              {type === 'gfpgan' && strength !== undefined && (
                <MetadataItem
                  label="Fix faces strength"
                  value={strength}
                  onClick={() => dispatch(setGfpganStrength(strength))}
                />
              )}
              {type === 'esrgan' && scale !== undefined && (
                <MetadataItem
                  label="Upscaling scale"
                  value={scale}
                  onClick={() => dispatch(setUpscalingLevel(scale))}
                />
              )}
              {type === 'esrgan' && strength !== undefined && (
                <MetadataItem
                  label="Upscaling strength"
                  value={strength}
                  onClick={() => dispatch(setUpscalingStrength(strength))}
                />
              )}
              {prompt && (
                <MetadataItem
                  label="Prompt"
                  labelPosition="top"
                  value={promptToString(prompt)}
                  onClick={() => dispatch(setPrompt(prompt))}
                />
              )}
              {seed !== undefined && (
                <MetadataItem
                  label="Seed"
                  value={seed}
                  onClick={() => dispatch(setSeed(seed))}
                />
              )}
              {sampler && (
                <MetadataItem
                  label="Sampler"
                  value={sampler}
                  onClick={() => dispatch(setSampler(sampler))}
                />
              )}
              {steps && (
                <MetadataItem
                  label="Steps"
                  value={steps}
                  onClick={() => dispatch(setSteps(steps))}
                />
              )}
              {cfg_scale !== undefined && (
                <MetadataItem
                  label="CFG scale"
                  value={cfg_scale}
                  onClick={() => dispatch(setCfgScale(cfg_scale))}
                />
              )}
              {variations && variations.length > 0 && (
                <MetadataItem
                  label="Seed-weight pairs"
                  value={seedWeightsToString(variations)}
                  onClick={() =>
                    dispatch(setSeedWeights(seedWeightsToString(variations)))
                  }
                />
              )}
              {seamless && (
                <MetadataItem
                  label="Seamless"
                  value={seamless}
                  onClick={() => dispatch(setWidth(seamless))}
                />
              )}
              {width && (
                <MetadataItem
                  label="Width"
                  value={width}
                  onClick={() => dispatch(setWidth(width))}
                />
              )}
              {height && (
                <MetadataItem
                  label="Height"
                  value={height}
                  onClick={() => dispatch(setHeight(height))}
                />
              )}
              {init_image_path && (
                <MetadataItem
                  label="Initial image"
                  value={init_image_path}
                  isLink
                  onClick={() => dispatch(setInitialImagePath(init_image_path))}
                />
              )}
              {mask_image_path && (
                <MetadataItem
                  label="Mask image"
                  value={mask_image_path}
                  isLink
                  onClick={() => dispatch(setMaskPath(mask_image_path))}
                />
              )}
              {type === 'img2img' && strength && (
                <MetadataItem
                  label="Image to image strength"
                  value={strength}
                  onClick={() => dispatch(setImg2imgStrength(strength))}
                />
              )}
              {fit && (
                <MetadataItem
                  label="Image to image fit"
                  value={fit}
                  onClick={() => dispatch(setShouldFitToWidthHeight(fit))}
                />
              )}
              {postprocessing && postprocessing.length > 0 && (
                <>
                  <Heading size={'sm'}>Postprocessing</Heading>
                  {postprocessing.map(
                    (
                      postprocess: InvokeAI.PostProcessedImageMetadata,
                      i: number
                    ) => {
                      if (postprocess.type === 'esrgan') {
                        const { scale, strength } = postprocess;
                        return (
                          <Flex
                            key={i}
                            pl={'2rem'}
                            gap={1}
                            direction={'column'}
                          >
                            <Text size={'md'}>{`${
                              i + 1
                            }: Upscale (ESRGAN)`}</Text>
                            <MetadataItem
                              label="Scale"
                              value={scale}
                              onClick={() => dispatch(setUpscalingLevel(scale))}
                            />
                            <MetadataItem
                              label="Strength"
                              value={strength}
                              onClick={() =>
                                dispatch(setUpscalingStrength(strength))
                              }
                            />
                          </Flex>
                        );
                      } else if (postprocess.type === 'gfpgan') {
                        const { strength } = postprocess;
                        return (
                          <Flex
                            key={i}
                            pl={'2rem'}
                            gap={1}
                            direction={'column'}
                          >
                            <Text size={'md'}>{`${
                              i + 1
                            }: Face restoration (GFPGAN)`}</Text>

                            <MetadataItem
                              label="Strength"
                              value={strength}
                              onClick={() =>
                                dispatch(setGfpganStrength(strength))
                              }
                            />
                          </Flex>
                        );
                      }
                    }
                  )}
                </>
              )}
              <Flex gap={2} direction={'column'}>
                <Flex gap={2}>
                  <Tooltip label={`Copy metadata JSON`}>
                    <IconButton
                      aria-label="Copy metadata JSON"
                      icon={<FaCopy />}
                      size={'xs'}
                      variant={'ghost'}
                      fontSize={14}
                      onClick={() =>
                        navigator.clipboard.writeText(metadataJSON)
                      }
                    />
                  </Tooltip>
                  <Text fontWeight={'semibold'}>Metadata JSON:</Text>
                </Flex>
                <div className={'image-json-viewer'}>
                  <pre>{metadataJSON}</pre>
                </div>
              </Flex>
            </>
          ) : (
            <Center width={'100%'} pt={10}>
              <Text fontSize={'lg'} fontWeight="semibold">
                No metadata available
              </Text>
            </Center>
          )}
        </Flex>
      </div>
    );
  },
  memoEqualityCheck
);

export default ImageMetadataViewer;
