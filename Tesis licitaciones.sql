CREATE TABLE `staging_documents` (
  `doc_id` bigserial PRIMARY KEY,
  `lic_id` text,
  `source_name` text,
  `source_ext` text,
  `doc_tag` text,
  `doc_order` int,
  `sha256` text,
  `chunk_count` bigint,
  `created_at` timestamptz,
  `updated_at` timestamptz
);

CREATE TABLE `staging_chunks` (
  `chunk_id` text PRIMARY KEY,
  `lic_id` text,
  `doc_id` bigint,
  `doc_chunk_index` int,
  `lic_chunk_index` int,
  `text` text,
  `created_at` timestamptz
);

CREATE TABLE `staging_licitations` (
  `lic_id` text PRIMARY KEY,
  `chunk_count` bigint,
  `document_count` bigint,
  `first_created_at` timestamptz,
  `last_created_at` timestamptz
);

CREATE TABLE `staging_doc_section_hits` (
  `doc_id` bigint,
  `lic_id` text,
  `section` text,
  `snippet` text,
  PRIMARY KEY (`doc_id`, `lic_id`, `section`)
);

CREATE TABLE `staging_document_text_samples` (
  `doc_id` bigint,
  `lic_id` text,
  `sample_text` text,
  PRIMARY KEY (`doc_id`, `lic_id`)
);

CREATE TABLE `public_licitacion` (
  `id` serial PRIMARY KEY,
  `entidad` varchar(255) NOT NULL,
  `objeto` text,
  `cuantia` numeric(18,2),
  `modalidad` varchar(255),
  `numero` varchar(255),
  `estado` varchar(100),
  `fecha_public` date,
  `ubicacion` varchar(255),
  `act_econ` varchar(255),
  `enlace` text,
  `portal_origen` varchar(255),
  `texto_indexado` text
);

CREATE TABLE `public_licitacion_keymap` (
  `licitacion_id` int PRIMARY KEY,
  `lic_ext_id` text UNIQUE
);

CREATE TABLE `public_licitacion_chunk` (
  `id` bigserial PRIMARY KEY,
  `licitacion_id` int NOT NULL,
  `chunk_idx` int NOT NULL,
  `chunk_text` text,
  `embedding` text COMMENT 'En PG puede guardarse el crudo opcional',
  `embedding_vec` text COMMENT 'En PG real es vector; aquí se muestra como TEXT'
);

CREATE TABLE `public_flags` (
  `id` serial PRIMARY KEY,
  `codigo` varchar(10) UNIQUE,
  `nombre` varchar(255),
  `descripcion` text
);

CREATE TABLE `public_flags_licitaciones` (
  `id` serial PRIMARY KEY,
  `licitacion_id` int,
  `flag_id` int,
  `valor` boolean,
  `fecha_detectado` timestamp,
  `comentario` text,
  `fuente` text
);

CREATE TABLE `public_flags_log` (
  `id` serial PRIMARY KEY,
  `flags_licitaciones_id` int,
  `cambio` text,
  `fecha` timestamp,
  `usuario` text
);

CREATE INDEX `public_licitacion_index_0` ON `public_licitacion` (`entidad`);

CREATE INDEX `public_licitacion_index_1` ON `public_licitacion` (`estado`);

CREATE INDEX `public_licitacion_index_2` ON `public_licitacion` (`fecha_public`);

CREATE UNIQUE INDEX `public_licitacion_chunk_index_3` ON `public_licitacion_chunk` (`licitacion_id`, `chunk_idx`);

CREATE UNIQUE INDEX `uq_flags_licitacion_flag` ON `public_flags_licitaciones` (`licitacion_id`, `flag_id`);

CREATE INDEX `public_flags_licitaciones_index_5` ON `public_flags_licitaciones` (`licitacion_id`, `flag_id`);

CREATE INDEX `public_flags_licitaciones_index_6` ON `public_flags_licitaciones` (`fecha_detectado`);

CREATE INDEX `public_flags_log_index_7` ON `public_flags_log` (`flags_licitaciones_id`);

CREATE INDEX `public_flags_log_index_8` ON `public_flags_log` (`fecha`);

ALTER TABLE `staging_chunks` ADD FOREIGN KEY (`doc_id`) REFERENCES `staging_documents` (`doc_id`);

ALTER TABLE `staging_doc_section_hits` ADD FOREIGN KEY (`doc_id`) REFERENCES `staging_documents` (`doc_id`);

ALTER TABLE `staging_document_text_samples` ADD FOREIGN KEY (`doc_id`) REFERENCES `staging_documents` (`doc_id`);

ALTER TABLE `public_licitacion_keymap` ADD FOREIGN KEY (`licitacion_id`) REFERENCES `public_licitacion` (`id`);

ALTER TABLE `public_licitacion_chunk` ADD FOREIGN KEY (`licitacion_id`) REFERENCES `public_licitacion` (`id`);

ALTER TABLE `public_flags_licitaciones` ADD FOREIGN KEY (`licitacion_id`) REFERENCES `public_licitacion` (`id`);

ALTER TABLE `public_flags_licitaciones` ADD FOREIGN KEY (`flag_id`) REFERENCES `public_flags` (`id`);

ALTER TABLE `public_flags_log` ADD FOREIGN KEY (`flags_licitaciones_id`) REFERENCES `public_flags_licitaciones` (`id`);
