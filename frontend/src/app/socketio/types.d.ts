/**
 * Interfaces used by the socketio middleware.
 */

export declare interface ServerGenerationResult {
  url: string;
  metadata: { [key: string]: any };
}

export declare interface ServerESRGANResult {
  url: string;
  uuid: string;
  metadata: { [key: string]: any };
}

export declare interface ServerGFPGANResult {
  url: string;
  uuid: string;
  metadata: { [key: string]: any };
}

export declare interface ServerIntermediateResult {
  url: string;
  metadata: { [key: string]: any };
}

export declare interface ServerError {
  message: string;
  additionalData?: string;
}

export declare interface ServerGalleryImages {
  images: Array<{
    path: string;
    metadata: { [key: string]: any };
  }>;
}

export declare interface ServerImageUrlAndUuid {
  uuid: string;
  url: string;
}

export declare interface ServerImageUrl {
  url: string;
}
