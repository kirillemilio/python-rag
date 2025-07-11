# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: proto/text/text_chunks_response.proto, proto/text/text_document_add_request.proto, proto/text/text_document_add_response.proto, proto/text/text_embedding_request.proto, proto/text/text_embedding_response.proto
# plugin: python-betterproto2
# This file has been @generated

__all__ = (
	'DocumentChunk',
	'TextChunksResponse',
	'TextDocumentAddRequest',
	'TextDocumentAddResponse',
	'TextRequest',
	'TextResponse',
)

from dataclasses import dataclass

import betterproto2

from ..message_pool import default_message_pool

betterproto2.check_compiler_version('0.6.0')


@dataclass(eq=False, repr=False)
class DocumentChunk(betterproto2.Message):
	document_id: 'str' = betterproto2.field(2, betterproto2.TYPE_STRING)

	content: 'str' = betterproto2.field(3, betterproto2.TYPE_STRING)

	source: 'str' = betterproto2.field(4, betterproto2.TYPE_STRING)

	collection: 'str' = betterproto2.field(5, betterproto2.TYPE_STRING)

	tags: 'list[str]' = betterproto2.field(6, betterproto2.TYPE_STRING, repeated=True)

	embedding: 'list[float]' = betterproto2.field(7, betterproto2.TYPE_FLOAT, repeated=True)

	created_at: 'float' = betterproto2.field(8, betterproto2.TYPE_FLOAT)


default_message_pool.register_message('text', 'DocumentChunk', DocumentChunk)


@dataclass(eq=False, repr=False)
class TextChunksResponse(betterproto2.Message):
	request_id: 'int' = betterproto2.field(1, betterproto2.TYPE_INT64)

	chunks: 'list[DocumentChunk]' = betterproto2.field(2, betterproto2.TYPE_MESSAGE, repeated=True)


default_message_pool.register_message('text', 'TextChunksResponse', TextChunksResponse)


@dataclass(eq=False, repr=False)
class TextDocumentAddRequest(betterproto2.Message):
	request_id: 'int' = betterproto2.field(1, betterproto2.TYPE_INT64)

	text: 'str' = betterproto2.field(2, betterproto2.TYPE_STRING)

	source: 'str' = betterproto2.field(3, betterproto2.TYPE_STRING)

	tags: 'list[str]' = betterproto2.field(4, betterproto2.TYPE_STRING, repeated=True)

	models: 'list[str]' = betterproto2.field(5, betterproto2.TYPE_STRING, repeated=True)


default_message_pool.register_message('text', 'TextDocumentAddRequest', TextDocumentAddRequest)


@dataclass(eq=False, repr=False)
class TextDocumentAddResponse(betterproto2.Message):
	request_id: 'int' = betterproto2.field(1, betterproto2.TYPE_INT64)

	document_id: 'str' = betterproto2.field(2, betterproto2.TYPE_STRING)

	created_at: 'float' = betterproto2.field(3, betterproto2.TYPE_FLOAT)


default_message_pool.register_message('text', 'TextDocumentAddResponse', TextDocumentAddResponse)


@dataclass(eq=False, repr=False)
class TextRequest(betterproto2.Message):
	request_id: 'int' = betterproto2.field(1, betterproto2.TYPE_INT64)

	text: 'str' = betterproto2.field(2, betterproto2.TYPE_STRING)

	source: 'str' = betterproto2.field(3, betterproto2.TYPE_STRING)

	tags: 'list[str]' = betterproto2.field(4, betterproto2.TYPE_STRING, repeated=True)

	model: 'str' = betterproto2.field(5, betterproto2.TYPE_STRING)


default_message_pool.register_message('text', 'TextRequest', TextRequest)


@dataclass(eq=False, repr=False)
class TextResponse(betterproto2.Message):
	request_id: 'int' = betterproto2.field(1, betterproto2.TYPE_INT64)

	model: 'str' = betterproto2.field(3, betterproto2.TYPE_STRING)

	embeddings: 'list[float]' = betterproto2.field(4, betterproto2.TYPE_FLOAT, repeated=True)


default_message_pool.register_message('text', 'TextResponse', TextResponse)
